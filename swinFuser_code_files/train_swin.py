import argparse
import json
import os
from tqdm import tqdm
import signal
import sys
import glob
import re
import gc
import psutil

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data, lidar_bev_cam_correspondences

import pathlib
import datetime
from torch.distributed.elastic.multiprocessing.errors import record
import random
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp

from diskcache import Cache

# ============================================================================
# ROBUST TRAINING FOR SWIN-PTT BACKBONE
# Features:
# - Automatic checkpoint resume (PM2/auto_restart_training.sh compatible)
# - OOM prevention with memory monitoring
# - Graceful shutdown on signals (SIGTERM, SIGINT)
# - Connection interrupt recovery
# - Gradient checkpointing for memory efficiency
# - INCOMPLETE CHECKPOINT CLEANUP - deletes partial saves on restart
# ============================================================================

class GracefulKiller:
    """Handle shutdown signals gracefully to save checkpoint before exit."""
    kill_now = False
    
    def __init__(self, trainer=None):
        self.trainer = trainer
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        # SIGUSR1 for manual checkpoint trigger
        signal.signal(signal.SIGUSR1, self.save_checkpoint)
    
    def exit_gracefully(self, signum, frame):
        print(f"\n[SIGNAL] Received signal {signum}. Saving checkpoint before exit...")
        GracefulKiller.kill_now = True
        if self.trainer is not None:
            self.trainer.save_emergency_checkpoint()
    
    def save_checkpoint(self, signum, frame):
        print(f"\n[SIGNAL] Manual checkpoint save triggered (SIGUSR1)")
        if self.trainer is not None:
            self.trainer.save()


class MemoryMonitor:
    """Monitor GPU and CPU memory to prevent OOM."""
    
    def __init__(self, gpu_threshold=0.90, cpu_threshold=0.90):
        self.gpu_threshold = gpu_threshold
        self.cpu_threshold = cpu_threshold
    
    def get_gpu_memory_usage(self, device_id=0):
        """Get GPU memory usage as a fraction."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
            return allocated / total
        return 0.0
    
    def get_cpu_memory_usage(self):
        """Get CPU memory usage as a fraction."""
        return psutil.virtual_memory().percent / 100.0
    
    def is_memory_critical(self, device_id=0):
        """Check if memory usage is critical."""
        gpu_usage = self.get_gpu_memory_usage(device_id)
        cpu_usage = self.get_cpu_memory_usage()
        return gpu_usage > self.gpu_threshold or cpu_usage > self.cpu_threshold
    
    def clear_cache(self):
        """Aggressive memory cleanup."""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def log_memory_status(self, device_id=0, prefix=""):
        """Log current memory status."""
        gpu_usage = self.get_gpu_memory_usage(device_id) * 100
        cpu_usage = self.get_cpu_memory_usage() * 100
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
        print(f"{prefix}GPU Memory: {gpu_usage:.1f}% (Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB) | CPU Memory: {cpu_usage:.1f}%")


def cleanup_incomplete_checkpoints(logdir):
    """
    Clean up incomplete checkpoints where model exists but optimizer doesn't (or vice versa).
    Also cleans up .tmp files from interrupted saves.
    Returns the epoch of the last COMPLETE checkpoint.
    """
    if not os.path.exists(logdir):
        return None
    
    print(f"[CLEANUP] Scanning {logdir} for incomplete checkpoints...")
    
    # Clean up any .tmp files first (interrupted atomic saves)
    tmp_files = glob.glob(os.path.join(logdir, "*.tmp"))
    for tmp_file in tmp_files:
        try:
            os.remove(tmp_file)
            print(f"[CLEANUP] Removed incomplete temp file: {tmp_file}")
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not remove {tmp_file}: {e}")
    
    # Find all model files (both normal and emergency)
    model_files = glob.glob(os.path.join(logdir, "model_*.pth"))
    emergency_model_files = glob.glob(os.path.join(logdir, "emergency_model_*.pth"))
    
    incomplete_epochs = []
    complete_epochs = []
    
    # Check normal checkpoints
    for model_file in model_files:
        match = re.search(r'model_(\d+)\.pth', model_file)
        if match:
            epoch = int(match.group(1))
            optimizer_file = os.path.join(logdir, f"optimizer_{epoch}.pth")
            
            if not os.path.exists(optimizer_file):
                incomplete_epochs.append(('normal', epoch, model_file, optimizer_file))
            else:
                # Verify both files are valid (not corrupted/empty)
                try:
                    if os.path.getsize(model_file) < 1000:  # Less than 1KB is suspicious
                        incomplete_epochs.append(('normal', epoch, model_file, optimizer_file))
                    elif os.path.getsize(optimizer_file) < 1000:
                        incomplete_epochs.append(('normal', epoch, model_file, optimizer_file))
                    else:
                        complete_epochs.append(('normal', epoch, model_file))
                except Exception as e:
                    incomplete_epochs.append(('normal', epoch, model_file, optimizer_file))
    
    # Check for orphaned optimizer files (optimizer without model)
    optimizer_files = glob.glob(os.path.join(logdir, "optimizer_*.pth"))
    for opt_file in optimizer_files:
        match = re.search(r'optimizer_(\d+)\.pth', opt_file)
        if match:
            epoch = int(match.group(1))
            model_file = os.path.join(logdir, f"model_{epoch}.pth")
            if not os.path.exists(model_file):
                # Orphaned optimizer file
                try:
                    os.remove(opt_file)
                    print(f"[CLEANUP] Removed orphaned optimizer file: {opt_file}")
                except Exception as e:
                    print(f"[CLEANUP] Warning: Could not remove {opt_file}: {e}")
    
    # Check emergency checkpoints
    for model_file in emergency_model_files:
        match = re.search(r'emergency_model_(\d+)\.pth', model_file)
        if match:
            epoch = int(match.group(1))
            optimizer_file = os.path.join(logdir, f"emergency_optimizer_{epoch}.pth")
            
            if not os.path.exists(optimizer_file):
                incomplete_epochs.append(('emergency', epoch, model_file, optimizer_file))
            else:
                # Verify both files are valid
                try:
                    if os.path.getsize(model_file) < 1000:
                        incomplete_epochs.append(('emergency', epoch, model_file, optimizer_file))
                    elif os.path.getsize(optimizer_file) < 1000:
                        incomplete_epochs.append(('emergency', epoch, model_file, optimizer_file))
                    else:
                        complete_epochs.append(('emergency', epoch, model_file))
                except Exception as e:
                    incomplete_epochs.append(('emergency', epoch, model_file, optimizer_file))
    
    # Check for orphaned emergency optimizer files
    emergency_opt_files = glob.glob(os.path.join(logdir, "emergency_optimizer_*.pth"))
    for opt_file in emergency_opt_files:
        match = re.search(r'emergency_optimizer_(\d+)\.pth', opt_file)
        if match:
            epoch = int(match.group(1))
            model_file = os.path.join(logdir, f"emergency_model_{epoch}.pth")
            if not os.path.exists(model_file):
                try:
                    os.remove(opt_file)
                    print(f"[CLEANUP] Removed orphaned emergency optimizer file: {opt_file}")
                except Exception as e:
                    print(f"[CLEANUP] Warning: Could not remove {opt_file}: {e}")
    
    # Remove incomplete checkpoints
    for checkpoint_type, epoch, model_file, optimizer_file in incomplete_epochs:
        print(f"[CLEANUP] Found incomplete {checkpoint_type} checkpoint at epoch {epoch}")
        
        # Remove model file if exists
        if os.path.exists(model_file):
            try:
                os.remove(model_file)
                print(f"[CLEANUP] Removed incomplete model: {model_file}")
            except Exception as e:
                print(f"[CLEANUP] Warning: Could not remove {model_file}: {e}")
        
        # Remove optimizer file if exists
        if os.path.exists(optimizer_file):
            try:
                os.remove(optimizer_file)
                print(f"[CLEANUP] Removed incomplete optimizer: {optimizer_file}")
            except Exception as e:
                print(f"[CLEANUP] Warning: Could not remove {optimizer_file}: {e}")
    
    # Find the latest complete checkpoint
    if complete_epochs:
        # Sort by epoch, prefer normal checkpoints over emergency
        complete_epochs.sort(key=lambda x: (x[1], x[0] == 'normal'), reverse=True)
        latest_type, latest_epoch, latest_file = complete_epochs[0]
        print(f"[CLEANUP] Latest complete checkpoint: {latest_type} epoch {latest_epoch}")
        return latest_epoch, latest_file, latest_type
    
    print("[CLEANUP] No complete checkpoints found")
    return None


def find_latest_checkpoint(logdir):
    """Find the latest COMPLETE checkpoint in the log directory."""
    if not os.path.exists(logdir):
        return None, 0
    
    # First, clean up incomplete checkpoints
    cleanup_result = cleanup_incomplete_checkpoints(logdir)
    
    if cleanup_result is None:
        return None, 0
    
    latest_epoch, latest_file, checkpoint_type = cleanup_result
    return latest_file, latest_epoch


def save_training_state(logdir, epoch, extra_info=None):
    """Save training state for PM2 restart."""
    state = {
        'last_completed_epoch': epoch,
        'timestamp': datetime.datetime.now().isoformat(),
        'extra_info': extra_info or {}
    }
    state_file = os.path.join(logdir, 'training_state.json')
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def load_training_state(logdir):
    """Load training state for resume."""
    state_file = os.path.join(logdir, 'training_state.json')
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return None


# Records error and tracebacks in case of failure
@record
def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='swin_ptt', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=41, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for one GPU. When training with multiple GPUs the effective batch size will be batch_size*num_gpus')
    parser.add_argument('--logdir', type=str, default='/home/group17/transfuser/logs', help='Directory to log data to.')
    parser.add_argument('--load_file', type=str, default=None, help='ckpt to load.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start with. Useful when continuing trainings via load_file.')
    parser.add_argument('--setting', type=str, default='all', help='What training setting to use. Options: '
                                                                   'all: Train on all towns no validation data. '
                                                                   '02_05_withheld: Do not train on Town 02 and Town 05. Use the data as validation data.')
    parser.add_argument('--root_dir', type=str, default='/home/group17/transfuser', help='Root directory of your training data')
    parser.add_argument('--schedule', type=int, default=1,
                        help='Whether to train with a learning rate schedule. 1 = True')
    parser.add_argument('--schedule_reduce_epoch_01', type=int, default=30,
                        help='Epoch at which to reduce the lr by a factor of 10 the first time. Only used with --schedule 1')
    parser.add_argument('--schedule_reduce_epoch_02', type=int, default=40,
                        help='Epoch at which to reduce the lr by a factor of 10 the second time. Only used with --schedule 1')
    
    # ===== SWIN-PTT SPECIFIC =====
    parser.add_argument('--backbone', type=str, default='swin_ptt',
                        help='Which Fusion backbone to use. Options: transFuser, late_fusion, latentTF, geometric_fusion, swin_ptt')
    parser.add_argument('--image_architecture', type=str, default='resnet34',
                        help='Which architecture to use for the image branch. resnet34, resnet18, regnety_032 etc.')
    parser.add_argument('--lidar_architecture', type=str, default='resnet18',
                        help='Which architecture to use for the lidar branch. resnet18, resnet34, regnety_032 etc.')
    
    parser.add_argument('--use_velocity', type=int, default=1,
                        help='Whether to use the velocity input. Expected values are 0:False, 1:True')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers used in the transfuser')
    parser.add_argument('--wp_only', type=int, default=0,
                        help='Valid values are 0, 1. 1 = using only the wp loss; 0= using all losses')
    parser.add_argument('--use_target_point_image', type=int, default=1,
                        help='Valid values are 0, 1. 1 = using target point in the LiDAR0; 0 = dont do it')
    parser.add_argument('--use_point_pillars', type=int, default=0,
                        help='Whether to use the point_pillar lidar encoder instead of voxelization. 0:False, 1:True')
    parser.add_argument('--parallel_training', type=int, default=1,
                        help='If this is true/1 you need to launch the train.py script with CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=123456780 --rdzv_backend=c10d train.py '
                             ' the code will be parallelized across GPUs. If set to false/0, you launch the script with python train.py and only 1 GPU will be used.')
    parser.add_argument('--val_every', type=int, default=5, help='At which epoch frequency to validate.')
    parser.add_argument('--no_bev_loss', type=int, default=0, help='If set to true the BEV loss will not be trained. 0: Train normally, 1: set training weight for BEV to 0')
    parser.add_argument('--sync_batch_norm', type=int, default=0, help='0: Compute batch norm for each GPU independently, 1: Synchronize Batch norms accross GPUs. Only use with --parallel_training 1')
    parser.add_argument('--zero_redundancy_optimizer', type=int, default=0, help='0: Normal AdamW Optimizer, 1: Use Zero Reduncdancy Optimizer to reduce memory footprint. Only use with --parallel_training 1')
    parser.add_argument('--use_disk_cache', type=int, default=0, help='0: Do not cache the dataset 1: Cache the dataset on the disk pointed to by the SCRATCH enironment variable. Useful if the dataset is stored on slow HDDs and can be temporarily stored on faster SSD storage.')
    
    # ROBUST TRAINING ARGUMENTS
    parser.add_argument('--auto_resume', type=int, default=1, help='1: Automatically resume from latest checkpoint, 0: Start fresh or use load_file')
    parser.add_argument('--gradient_checkpointing', type=int, default=0, help='1: Enable gradient checkpointing to save memory, 0: Disabled')
    parser.add_argument('--memory_efficient', type=int, default=1, help='1: Enable memory efficient mode (clear cache, monitor memory), 0: Disabled')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--gpu_memory_threshold', type=float, default=0.90, help='GPU memory threshold to trigger cleanup (0.0-1.0)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps (helps with OOM)')
    parser.add_argument('--cleanup_incomplete', type=int, default=1, help='1: Clean up incomplete checkpoints on startup, 0: Skip cleanup')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)
    parallel = bool(args.parallel_training)

    # Initialize memory monitor
    memory_monitor = MemoryMonitor(gpu_threshold=args.gpu_memory_threshold)

    if(bool(args.use_disk_cache) == True):
        if (parallel == True):
            tmp_folder = str(os.environ.get('SCRATCH'))
            print("Tmp folder for dataset cache: ", tmp_folder)
            tmp_folder = tmp_folder + "/dataset_cache"
            shared_dict = Cache(directory=tmp_folder, size_limit=int(768 * 1024 ** 3))
        else:
            shared_dict = Cache(size_limit=int(768 * 1024 ** 3))
    else:
        shared_dict = None

    # Use torchrun for starting because it has proper error handling
    if(parallel == True):
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}")

        device = torch.device('cuda:{}'.format(local_rank))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

        # Increased timeout for connection recovery
        torch.distributed.init_process_group(
            backend='nccl', 
            init_method='env://', 
            world_size=world_size, 
            rank=rank,
            timeout=datetime.timedelta(minutes=30)  # Increased timeout for connection issues
        )

        torch.distributed.barrier(device_ids=[local_rank])
    else:
        rank       = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda:{}'.format(local_rank))

    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    # Configure config
    config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)
    config.use_target_point_image = bool(args.use_target_point_image)
    config.n_layer = args.n_layer
    config.use_point_pillars = bool(args.use_point_pillars)
    config.backbone = args.backbone
    if(bool(args.no_bev_loss)):
        index_bev = config.detailed_losses.index("loss_bev")
        config.detailed_losses_weights[index_bev] = 0.0

    # ===== CREATE SWIN-PTT MODEL =====
    print(f"\n{'='*60}")
    print(f"[MODEL] Creating Swin-PTT backbone")
    print(f"[MODEL] Image architecture: {args.image_architecture}")
    print(f"[MODEL] LiDAR architecture: {args.lidar_architecture}")
    print(f"[MODEL] Use velocity: {bool(args.use_velocity)}")
    print(f"{'='*60}\n")
    
    model = LidarCenterNet(config, device, args.backbone, args.image_architecture, args.lidar_architecture, bool(args.use_velocity))

    # Enable gradient checkpointing if requested (saves memory)
    if bool(args.gradient_checkpointing):
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
            print("[MEMORY] Gradient checkpointing enabled")
        else:
            print("[MEMORY] Warning: Model does not support gradient checkpointing")

    if (parallel == True):
        if(bool(args.sync_batch_norm) == True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank, 
            broadcast_buffers=False, 
            find_unused_parameters=False
        )

    model.cuda(device=device)

    if ((bool(args.zero_redundancy_optimizer) == True) and (parallel == True)):
        optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=optim.AdamW, lr=args.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)

    # Data
    train_set = CARLA_Data(root=config.train_data, config=config, shared_dict=shared_dict)
    val_set   = CARLA_Data(root=config.val_data,   config=config, shared_dict=shared_dict)

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())

    # Reduce num_workers if memory is constrained - use 0 for shared HPC with limited RAM
    num_workers = 4 if not bool(args.memory_efficient) else 0

    if(parallel == True):
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, num_replicas=world_size, rank=rank)
        sampler_val   = torch.utils.data.distributed.DistributedSampler(val_set,   shuffle=True, num_replicas=world_size, rank=rank)
        dataloader_train = DataLoader(train_set, sampler=sampler_train, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=num_workers, pin_memory=True)
        dataloader_val   = DataLoader(val_set,   sampler=sampler_val,   batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=num_workers, pin_memory=True)
    else:
        dataloader_train = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)
        dataloader_val   = DataLoader(val_set,   shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)

    # Create logdir
    if ((not os.path.isdir(args.logdir)) and (rank == 0)):
        print('Created dir:', args.logdir, rank)
        os.makedirs(args.logdir, exist_ok=True)

    if(rank == 0):
        writer = SummaryWriter(log_dir=args.logdir)
        with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        writer = None

    # ========================================================================
    # AUTO-RESUME LOGIC WITH INCOMPLETE CHECKPOINT CLEANUP
    # ========================================================================
    start_epoch = args.start_epoch
    
    if bool(args.auto_resume) and args.load_file is None:
        # Clean up incomplete checkpoints and find the latest complete one
        if rank == 0 and bool(args.cleanup_incomplete):
            print("\n" + "="*60)
            print("[STARTUP] Checking for incomplete checkpoints...")
            print("="*60)
        
        # Try to find the latest COMPLETE checkpoint
        latest_checkpoint, latest_epoch = find_latest_checkpoint(args.logdir)
        
        if latest_checkpoint is not None:
            print(f"[AUTO-RESUME] Found complete checkpoint at epoch {latest_epoch}: {latest_checkpoint}")
            args.load_file = latest_checkpoint
            start_epoch = latest_epoch + 1
            print(f"[AUTO-RESUME] Will resume training from epoch {start_epoch}")
        else:
            print("[AUTO-RESUME] No complete checkpoints found. Starting from scratch.")

    if args.load_file is not None:
        print("=============load=================")
        print(f"Loading model from: {args.load_file}")
        
        # Handle both emergency and normal checkpoints
        checkpoint = torch.load(args.load_file, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Emergency checkpoint format
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', start_epoch) + 1  # Resume from NEXT epoch
            interrupted_epoch = checkpoint.get('interrupted_epoch', None)
            if interrupted_epoch is not None:
                print(f"[RESUME] Loaded emergency checkpoint: epoch {interrupted_epoch} was interrupted, will restart from epoch {start_epoch}")
            else:
                print(f"[RESUME] Loaded emergency checkpoint, resuming from epoch {start_epoch}")
        else:
            # Normal checkpoint format
            model.load_state_dict(checkpoint)
            optimizer_file = args.load_file.replace("model_", "optimizer_").replace("emergency_model_", "emergency_optimizer_")
            if os.path.exists(optimizer_file):
                optimizer.load_state_dict(torch.load(optimizer_file, map_location=device))

    # ========================================================================
    # CREATE TRAINER WITH MEMORY MONITORING
    # ========================================================================
    trainer = Engine(
        model=model, 
        optimizer=optimizer, 
        dataloader_train=dataloader_train, 
        dataloader_val=dataloader_val,
        args=args, 
        config=config, 
        writer=writer, 
        device=device, 
        rank=rank, 
        world_size=world_size,
        parallel=parallel, 
        cur_epoch=start_epoch,
        memory_monitor=memory_monitor,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Set up graceful shutdown handler
    killer = GracefulKiller(trainer=trainer)

    print(f"\n{'='*60}")
    print(f"[TRAINING] SWIN-PTT Training Configuration")
    print(f"[TRAINING] Starting from epoch {start_epoch} to {args.epochs}")
    print(f"[TRAINING] Auto-resume: {bool(args.auto_resume)}")
    print(f"[TRAINING] Cleanup incomplete checkpoints: {bool(args.cleanup_incomplete)}")
    print(f"[TRAINING] Memory efficient mode: {bool(args.memory_efficient)}")
    print(f"[TRAINING] Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"{'='*60}\n")

    # Log initial memory status
    if rank == 0 and bool(args.memory_efficient):
        memory_monitor.log_memory_status(local_rank, "[INIT] ")

    for epoch in range(trainer.cur_epoch, args.epochs):
        # Check for graceful shutdown
        if GracefulKiller.kill_now:
            print("[SHUTDOWN] Graceful shutdown requested. Exiting training loop.")
            break

        if(parallel == True):
            sampler_train.set_epoch(epoch)
        
        if ((epoch == args.schedule_reduce_epoch_01) or (epoch == args.schedule_reduce_epoch_02)) and (args.schedule == 1):
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.1
            print("Reduce learning rate by factor 10 to:", new_lr)
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        
        # Memory cleanup before each epoch
        if bool(args.memory_efficient):
            memory_monitor.clear_cache()
            if rank == 0:
                memory_monitor.log_memory_status(local_rank, f"[Epoch {epoch} Start] ")
        
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM ERROR] GPU out of memory at epoch {epoch}!")
                print("[OOM] Saving emergency checkpoint...")
                trainer.save_emergency_checkpoint()
                memory_monitor.clear_cache()
                
                # Try to continue with reduced batch processing
                print("[OOM] Attempting to continue with aggressive memory cleanup...")
                gc.collect()
                torch.cuda.empty_cache()
                
                # Re-raise to let the process restart
                raise
            else:
                raise

        if((args.setting != 'all') and (epoch % args.val_every == 0)):
            trainer.validate()

        # Save checkpoint
        should_save = (epoch % args.save_every == 0) or (epoch == args.epochs - 1)
        if should_save:
            if (parallel == True):
                if (bool(args.zero_redundancy_optimizer) == True):
                    optimizer.consolidate_state_dict(0)
                if (rank == 0):
                    trainer.save()
                    save_training_state(args.logdir, epoch)
            else:
                trainer.save()
                save_training_state(args.logdir, epoch)

        # Memory cleanup after each epoch
        if bool(args.memory_efficient):
            memory_monitor.clear_cache()

    # Final save
    if rank == 0:
        print("[TRAINING] Training completed successfully!")
        save_training_state(args.logdir, trainer.cur_epoch, {'status': 'completed'})


class Engine(object):
    """
    Engine that runs training with memory monitoring and emergency checkpointing.
    """

    def __init__(self, model, optimizer, dataloader_train, dataloader_val, args, config, writer, device, 
                 rank=0, world_size=1, parallel=False, cur_epoch=0, memory_monitor=None, gradient_accumulation_steps=1):
        self.cur_epoch = cur_epoch
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_val   = dataloader_val
        self.args = args
        self.config = config
        self.writer = writer
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.parallel = parallel
        self.vis_save_path = self.args.logdir + r'/visualizations'
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        if(self.config.debug == True):
            pathlib.Path(self.vis_save_path).mkdir(parents=True, exist_ok=True)

        self.detailed_losses = config.detailed_losses
        if self.args.wp_only:
            detailed_losses_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            detailed_losses_weights = config.detailed_losses_weights
        self.detailed_weights = {key: detailed_losses_weights[idx] for idx, key in enumerate(self.detailed_losses)}

    def load_data_compute_loss(self, data):
        # Move data to GPU
        rgb = data['rgb'].to(self.device, dtype=torch.float32)
        if self.config.multitask:
            depth = data['depth'].to(self.device, dtype=torch.float32)
            semantic = data['semantic'].squeeze(1).to(self.device, dtype=torch.long)
        else:
            depth = None
            semantic = None

        bev = data['bev'].to(self.device, dtype=torch.long)

        if (self.config.use_point_pillars == True):
            lidar = data['lidar_raw'].to(self.device, dtype=torch.float32)
            num_points = data['num_points'].to(self.device, dtype=torch.int32)
        else:
            lidar = data['lidar'].to(self.device, dtype=torch.float32)
            num_points = None

        label = data['label'].to(self.device, dtype=torch.float32)
        ego_waypoint = data['ego_waypoint'].to(self.device, dtype=torch.float32)

        target_point = data['target_point'].to(self.device, dtype=torch.float32)
        target_point_image = data['target_point_image'].to(self.device, dtype=torch.float32)

        ego_vel = data['speed'].to(self.device, dtype=torch.float32)

        # SWIN-PTT uses same interface as transFuser
        if ((self.args.backbone == 'transFuser') or (self.args.backbone == 'late_fusion') or 
            (self.args.backbone == 'latentTF') or (self.args.backbone == 'swin_ptt')):
            losses = self.model(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                           target_point_image=target_point_image,
                           ego_vel=ego_vel.reshape(-1, 1), bev=bev,
                           label=label, save_path=self.vis_save_path,
                           depth=depth, semantic=semantic, num_points=num_points)
        elif (self.args.backbone == 'geometric_fusion'):
            bev_points = data['bev_points'].long().to('cuda', dtype=torch.int64)
            cam_points = data['cam_points'].long().to('cuda', dtype=torch.int64)
            losses = self.model(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                           target_point_image=target_point_image,
                           ego_vel=ego_vel.reshape(-1, 1), bev=bev,
                           label=label, save_path=self.vis_save_path,
                           depth=depth, semantic=semantic, num_points=num_points,
                           bev_points=bev_points, cam_points=cam_points)
        else:
            raise ValueError("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF, swin_ptt")

        return losses

    def train(self):
        self.model.train()

        num_batches = 0
        loss_epoch = 0.0
        detailed_losses_epoch = {key: 0.0 for key in self.detailed_losses}
        self.cur_epoch += 1

        # Train loop with gradient accumulation
        self.optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(self.dataloader_train, desc=f"Epoch {self.cur_epoch}")
        for batch_idx, data in enumerate(pbar):
            # Check for graceful shutdown
            if GracefulKiller.kill_now:
                print(f"[SHUTDOWN] Saving checkpoint at batch {batch_idx}...")
                self.save_emergency_checkpoint()
                break
            
            losses = self.load_data_compute_loss(data)
            loss = torch.tensor(0.0).to(self.device, dtype=torch.float32)

            for key, value in losses.items():
                loss += self.detailed_weights[key] * value
                detailed_losses_epoch[key] += float(self.detailed_weights[key] * value.item())
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Memory cleanup if enabled
                if bool(self.args.memory_efficient) and batch_idx % 100 == 0:
                    self.memory_monitor.clear_cache()

            num_batches += 1
            loss_epoch += float(loss.item() * self.gradient_accumulation_steps)  # Rescale for logging
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}'})

        # Handle any remaining gradients
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, '')

    @torch.inference_mode()
    def validate(self):
        self.model.eval()

        num_batches = 0
        loss_epoch = 0.0
        detailed_val_losses_epoch = {key: 0.0 for key in self.detailed_losses}

        for data in tqdm(self.dataloader_val, desc="Validation"):
            losses = self.load_data_compute_loss(data)

            loss = torch.tensor(0.0).to(self.device, dtype=torch.float32)

            for key, value in losses.items():
                loss += self.detailed_weights[key] * value
                detailed_val_losses_epoch[key] += float(self.detailed_weights[key] * value.item())

            num_batches += 1
            loss_epoch += float(loss.item())

        self.log_losses(loss_epoch, detailed_val_losses_epoch, num_batches, 'val_')

    def log_losses(self, loss_epoch, detailed_losses_epoch, num_batches, prefix=''):
        if num_batches == 0:
            return
            
        loss_epoch = loss_epoch / num_batches
        for key, value in detailed_losses_epoch.items():
            detailed_losses_epoch[key] = value / num_batches

        gathered_detailed_losses = [None for _ in range(self.world_size)]
        gathered_loss = [None for _ in range(self.world_size)]

        if (self.parallel == True):
            torch.distributed.gather_object(obj=detailed_losses_epoch,
                                            object_gather_list=gathered_detailed_losses if self.rank == 0 else None, 
                                            dst=0)
            torch.distributed.gather_object(obj=loss_epoch, 
                                            object_gather_list=gathered_loss if self.rank == 0 else None,
                                            dst=0)
        else:
            gathered_detailed_losses[0] = detailed_losses_epoch
            gathered_loss[0] = loss_epoch
            
        if (self.rank == 0):
            aggregated_total_loss = sum(gathered_loss) / len(gathered_loss)
            self.writer.add_scalar(prefix + 'loss_total', aggregated_total_loss, self.cur_epoch)

            for key, value in detailed_losses_epoch.items():
                aggregated_value = 0.0
                for i in range(self.world_size):
                    aggregated_value += gathered_detailed_losses[i][key]

                aggregated_value = aggregated_value / self.world_size
                self.writer.add_scalar(prefix + key, aggregated_value, self.cur_epoch)

    def save(self):
        """Normal checkpoint save with atomic write to prevent corruption."""
        if self.rank == 0:
            model_path = os.path.join(self.args.logdir, 'model_%d.pth' % self.cur_epoch)
            optim_path = os.path.join(self.args.logdir, 'optimizer_%d.pth' % self.cur_epoch)

            # Atomic save: write to temp file, then rename
            model_tmp = model_path + '.tmp'
            optim_tmp = optim_path + '.tmp'

            torch.save(self.model.state_dict(), model_tmp)
            os.rename(model_tmp, model_path)

            torch.save(self.optimizer.state_dict(), optim_tmp)
            os.rename(optim_tmp, optim_path)

            print(f"[SAVE] Checkpoint saved at epoch {self.cur_epoch}")
            
            # Clean up emergency checkpoints since we saved successfully
            self._cleanup_emergency_checkpoints()
    
    def _cleanup_emergency_checkpoints(self):
        """Remove emergency checkpoints after successful save."""
        try:
            emergency_files = glob.glob(os.path.join(self.args.logdir, "emergency_*.pth"))
            emergency_tmp_files = glob.glob(os.path.join(self.args.logdir, "emergency_*.pth.tmp"))
            for f in emergency_files + emergency_tmp_files:
                os.remove(f)
                print(f"[CLEANUP] Removed emergency checkpoint: {f}")
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not remove emergency checkpoints: {e}")

    def save_emergency_checkpoint(self):
        """Save emergency checkpoint with atomic write to prevent corruption.

        Since cur_epoch is incremented at the START of train(), an interrupted
        epoch is not complete. We save cur_epoch - 1 (last completed epoch) so
        that on resume, training restarts from the interrupted epoch.
        """
        if self.rank == 0:
            # Save as last COMPLETED epoch, since current epoch is incomplete
            last_completed_epoch = max(0, self.cur_epoch - 1)
            emergency_path = os.path.join(self.args.logdir, f'emergency_model_{last_completed_epoch}.pth')
            emergency_optim_path = os.path.join(self.args.logdir, f'emergency_optimizer_{last_completed_epoch}.pth')

            checkpoint = {
                'epoch': last_completed_epoch,
                'interrupted_epoch': self.cur_epoch,  # For debugging: which epoch was interrupted
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'val_loss': self.val_loss,
                'bestval': self.bestval,
            }

            # Atomic save: write to temp file, then rename
            emergency_tmp = emergency_path + '.tmp'
            emergency_optim_tmp = emergency_optim_path + '.tmp'

            torch.save(checkpoint, emergency_tmp)
            os.rename(emergency_tmp, emergency_path)

            torch.save(self.optimizer.state_dict(), emergency_optim_tmp)
            os.rename(emergency_optim_tmp, emergency_optim_path)

            print(f"[EMERGENCY SAVE] Interrupted during epoch {self.cur_epoch}, saved as epoch {last_completed_epoch}: {emergency_path}")

            # Also save training state with last completed epoch
            save_training_state(self.args.logdir, last_completed_epoch, {'emergency': True, 'interrupted_epoch': self.cur_epoch})


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    mp.set_start_method('fork')
    print("Start method of multiprocessing:", mp.get_start_method())
    main()
