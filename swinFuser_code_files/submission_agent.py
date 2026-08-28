import os
import json
from copy import deepcopy

import cv2
import carla
from PIL import Image
from collections import deque

import torch
import numpy as np
import math

from leaderboard.autoagents import autonomous_agent
from model import LidarCenterNet
from config import GlobalConfig
from data import lidar_to_histogram_features, draw_target_point, lidar_bev_cam_correspondences

from shapely.geometry import Polygon

import itertools
import pathlib
SAVE_PATH = os.environ.get('SAVE_PATH')

if not SAVE_PATH:
    SAVE_PATH = None
else:
    pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

def get_entry_point():
    return 'HybridAgent'


class HybridAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False

        args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
        self.args = json.load(args_file)
        args_file.close()

        # setting machine to avoid loading files
        self.config = GlobalConfig(setting='eval')

        if ('sync_batch_norm' in self.args):
            self.config.sync_batch_norm = bool(self.args['sync_batch_norm'])
        if ('use_point_pillars' in self.args):
            self.config.use_point_pillars = self.args['use_point_pillars']
        if ('n_layer' in self.args):
            self.config.n_layer = self.args['n_layer']
        if ('use_target_point_image' in self.args):
            self.config.use_target_point_image = bool(self.args['use_target_point_image'])
        if ('use_velocity' in self.args):
            use_velocity = bool(self.args['use_velocity'])
        else:
            use_velocity = True

        if ('image_architecture' in self.args):
            image_architecture = self.args['image_architecture']
        else:
            image_architecture = 'resnet34'

        if ('lidar_architecture' in self.args):
            lidar_architecture = self.args['lidar_architecture']
        else:
            lidar_architecture = 'resnet18'

        if ('backbone' in self.args):
            self.backbone = self.args['backbone']  # Options 'geometric_fusion', 'transFuser', 'late_fusion', 'latentTF'
        else:
            self.backbone = 'transFuser'  # Options 'geometric_fusion', 'transFuser', 'late_fusion', 'latentTF'

        self.gps_buffer = deque(maxlen=self.config.gps_buffer_max_len) # Stores the last x updated gps signals.
        self.ego_model = EgoModel(dt=self.config.carla_frame_rate) # Bicycle model used for de-noising the GPS

        self.bb_buffer = deque(maxlen=1)

        # --- SWIFT-MPC ---
        from mpc_controller import SamplingMPC, obstacles_from_bbs
        self.use_swift_mpc = int(os.environ.get('SWIFT_MPC', '0')) == 1
        self._obstacles_from_bbs = obstacles_from_bbs
        self.prev_bbs_mpc = None   # always needed: the recovery FSM uses it even when MPC is off
        if self.use_swift_mpc:
            self.mpc = SamplingMPC(self.config)   # reads mpc_* from config if present

        # --- Recovery policy (deadlock FSM + low-speed MPC escape + 3-frame vel) ---
        import types as _types
        from recovery_fsm import RecoveryFSM, creep_safety_gate
        self._creep_safety_gate = creep_safety_gate
        from waypoint_safety import waypoint_safety_check, format_safety_attribution
        self._waypoint_safety_check = waypoint_safety_check
        self._format_safety_attribution = format_safety_attribution

        # ============================================================
        # SAFETY-XAI EVENT RECORDER (diagnostic only)
        # Saves real RGB/LiDAR/trajectory/obstacle/shield data for
        # publication figures without changing the shield decision.
        # ============================================================
        self.save_safety_xai = int(os.environ.get('SAVE_SAFETY_XAI', '1')) == 1
        self.safety_xai_dir = os.environ.get(
            'SAFETY_XAI_DIR',
            os.path.join(os.getcwd(), 'safety_xai_samples')
        )
        self.safety_xai_count = 0
        self.safety_xai_last_save_step = -1000000
        self.safety_xai_min_gap = int(os.environ.get('SAFETY_XAI_MIN_GAP', '20'))

        if self.save_safety_xai:
            os.makedirs(self.safety_xai_dir, exist_ok=True)

        print(
            "[SAFETY-XAI] recorder:",
            "ON" if self.save_safety_xai else "OFF",
            "dir=", self.safety_xai_dir,
            "min_gap=", self.safety_xai_min_gap,
            flush=True
        )
        self.use_waypoint_safety = (int(os.environ.get(
            'WP_SAFETY', '1' if getattr(self.config, 'use_waypoint_safety', False) else '0')) == 1)
        # --- Learned detection head (Option A: detect-and-brake) ---
        self.use_safety_head = (int(os.environ.get('SAFETY_HEAD', '0')) == 1)
        self.safety_head = None
        if self.use_safety_head:
            try:
                from safety_head_infer import SafetyHead
                ckpt = os.environ.get('SAFETY_HEAD_CKPT',
                                      getattr(self.config, 'safety_head_ckpt', 'detection_head.pt'))
                self.safety_head = SafetyHead(ckpt, device='cuda')
                self.safety_head_thresh = float(os.environ.get(
                    'SAFETY_HEAD_THRESH',
                    getattr(self.config, 'safety_head_thresh', 0.5)))
                print(f"[HEAD] loaded learned safety head from {ckpt}, "
                      f"threshold={self.safety_head_thresh}")
            except Exception as e:
                print(f"[HEAD] failed to load, disabling: {repr(e)}")
                self.use_safety_head = False

        # --- Level 1: trajectory checker (predicts OTHER vehicles' paths) ---
        # Catches CROSSING traffic the snapshot-based head misses: forecasts each
        # nearby vehicle ~2s ahead and brakes if its predicted path will cross ours.
        self.use_traj_check = (int(os.environ.get('TRAJ_CHECK', '0')) == 1)
        self.traj_checker = None
        if self.use_traj_check:
            try:
                from trajectory_check import TrajectoryChecker
                mckpt = os.environ.get('MOTION_CKPT',
                                       getattr(self.config, 'motion_ckpt',
                                               'motion_predictor_2s.pt'))
                self.traj_checker = TrajectoryChecker(
                    mckpt, device='cuda',
                    lon_margin=float(os.environ.get(
                        'TRAJ_LON_MARGIN',
                        getattr(self.config, 'traj_lon_margin', 2.5))),
                    lat_margin=float(os.environ.get(
                        'TRAJ_LAT_MARGIN',
                        getattr(self.config, 'traj_lat_margin', 1.6))))
                # same anti-over-braking guard as the head: don't freeze a crawling car
                self.traj_min_speed = float(os.environ.get(
                    'TRAJ_MIN_SPEED',
                    getattr(self.config, 'traj_min_speed', 1.5)))
                print(f"[TRAJ] loaded motion predictor from {mckpt} "
                      f"(horizon {self.traj_checker.fut} steps x 0.5s)")
            except Exception as e:
                print(f"[TRAJ] failed to load, disabling: {repr(e)}")
                self.use_traj_check = False
        from mpc_recovery_ext import steer_assist_lowspeed, obstacles_from_bbs_3frame
        # env override lets you A/B the FSM independently of config default
        self.use_recovery_fsm = (int(os.environ.get(
            'RECOVERY_FSM', '1' if getattr(self.config, 'use_recovery_fsm', False) else '0')) == 1)
        self.use_3frame_velocity = getattr(self.config, 'use_3frame_velocity', False)
        self.use_mpc_lowspeed_escape = getattr(self.config, 'use_mpc_lowspeed_escape', False)
        self.recovery = RecoveryFSM(self.config)
        self._obstacles_from_bbs_3frame = obstacles_from_bbs_3frame
        self.prev2_bbs_mpc = None                 # second-oldest bb set for 3-frame vel
        if self.use_swift_mpc:                    # bind low-speed escape onto the mpc
            self.mpc.steer_assist_lowspeed = _types.MethodType(
                lambda m, wp, obs, creep_v: steer_assist_lowspeed(m, wp, obs, creep_v),
                self.mpc)
        # --- end recovery setup ---
            print("[MPC] SWIFT-MPC ENABLED")
        # --- end SWIFT-MPC ---
        self.lidar_pos = self.config.lidar_pos  # x, y, z coordinates of the LiDAR position.
        self.iou_treshold_nms = self.config.iou_treshold_nms # Iou threshold used for Non Maximum suppression on the Bounding Box predictions.


        # Load model files
        self.nets = []
        self.model_count = 0 # Counts how many models are in our ensemble
        for file in os.listdir(path_to_conf_file):
            if file.endswith(".pth"):
                self.model_count += 1
                print(os.path.join(path_to_conf_file, file))
                net = LidarCenterNet(self.config, 'cuda', self.backbone, image_architecture, lidar_architecture, use_velocity)
                if(self.config.sync_batch_norm == True):
                    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net) # Model was trained with Sync. Batch Norm. Need to convert it otherwise parameters will load incorrectly.
                state_dict = torch.load(os.path.join(path_to_conf_file, file), map_location='cuda:0')
                #state_dict = {k[7:]: v for k, v in state_dict.items()} # Removes the .module coming from the Distributed Training. Remove this if you want to evaluate a model trained without DDP.
                def fix_state_dict_keys(state_dict):
                    """Fix checkpoint keys to match current model architecture"""
                    new_state_dict = {}
                    
                    for key, value in state_dict.items():
                        # Remove 'module.' prefix if present (from DDP training)
                        if key.startswith('module.'):
                            key = key[7:]
                        
                        # Map backbone components to _model wrapper
                        backbone_components = [
                            'image_encoder', 'lidar_encoder', 
                            'swin_fusion1', 'swin_fusion2', 'swin_fusion3', 'swin_fusion4',
                            'up_conv5', 'up_conv4', 'up_conv3', 'c5_conv',
                            'change_channel_conv_image', 'change_channel_conv_lidar',
                            'unified_pool', 'relu', 'upsample'
                        ]
                        
                        # Check if this key belongs to backbone components
                        needs_model_prefix = any(key.startswith(component) for component in backbone_components)
                        
                        if needs_model_prefix:
                            new_key = f"_model.{key}"
                        else:
                            # Keys like seg_decoder, depth_decoder, pred_bev, head, join, decoder, output
                            # remain as they are
                            new_key = key
                            
                        new_state_dict[new_key] = value
                    
                    return new_state_dict

                # Apply the fix
                state_dict = fix_state_dict_keys(state_dict)
                
                net.load_state_dict(state_dict, strict=True)
                net.cuda()
                net.eval()
                self.nets.append(net)


        self.stuck_detector = 0
        self.forced_move = 0
        if hasattr(self, 'recovery'):   # reset deadlock FSM per route
            self.recovery.reset()
            self.prev2_bbs_mpc = None

        self.use_lidar_safe_check = True
        self.aug_degrees = [0] # Test time data augmentation. Unused we only augment by 0 degree.
        self.steer_damping = self.config.steer_damping
        self.rgb_back = None #For debugging



    def _init(self):
        self._route_planner = RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        sensors = [
                    {
                        'type': 'sensor.camera.rgb',
                        'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z':self.config.camera_pos[2],
                        'roll': self.config.camera_rot_0[0], 'pitch': self.config.camera_rot_0[1], 'yaw': self.config.camera_rot_0[2],
                        'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                        'id': 'rgb_front'
                        },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z':self.config.camera_pos[2],
                        'roll': self.config.camera_rot_1[0], 'pitch': self.config.camera_rot_1[1], 'yaw': self.config.camera_rot_1[2],
                        'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                        'id': 'rgb_left'
                        },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z':self.config.camera_pos[2],
                        'roll': self.config.camera_rot_2[0], 'pitch': self.config.camera_rot_2[1], 'yaw': self.config.camera_rot_2[2],
                        'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                        'id': 'rgb_right'
                        },
                    {
                        'type': 'sensor.other.imu',
                        'x': 0.0, 'y': 0.0, 'z': 0.0,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'sensor_tick': self.config.carla_frame_rate,
                        'id': 'imu'
                        },
                    {
                        'type': 'sensor.other.gnss',
                        'x': 0.0, 'y': 0.0, 'z': 0.0,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'sensor_tick': 0.01,
                        'id': 'gps'
                        },
                    {
                        'type': 'sensor.speedometer',
                        'reading_frequency': self.config.carla_fps,
                        'id': 'speed'
                        }
                    ]
        if(SAVE_PATH != None): #Debug camera for visualizations
            sensors.append({
                            'type': 'sensor.camera.rgb',
                            'x': -4.5, 'y': 0.0, 'z':2.3,
                            'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
                            'width': 960, 'height': 480, 'fov': 100,
                            'id': 'rgb_back'
                            })

        if (self.backbone != 'latentTF'):  # LiDAR method
            sensors.append({
                            'type': 'sensor.lidar.ray_cast',
                            'x': self.lidar_pos[0], 'y': self.lidar_pos[1], 'z': self.lidar_pos[2],
                            'roll': self.config.lidar_rot[0], 'pitch': self.config.lidar_rot[1], 'yaw': self.config.lidar_rot[2],
                            'id': 'lidar'
                           })

        return sensors

    def tick(self, input_data):
        rgb = []
        for pos in ['left', 'front', 'right']:
            rgb_cam = 'rgb_' + pos
            rgb_pos = cv2.cvtColor(input_data[rgb_cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            rgb_pos = self.scale_crop(Image.fromarray(rgb_pos), self.config.scale, self.config.img_width, self.config.img_width, self.config.img_resolution[0], self.config.img_resolution[0])
            rgb.append(rgb_pos)
        rgb = np.concatenate(rgb, axis=1)

        if(SAVE_PATH != None): #Debug camera for visualizations
            # don't need buffer for it always use the latest one
            self.rgb_back = input_data["rgb_back"][1][:, :, :3]

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        if (np.isnan(compass) == True): # CARLA 0.9.10 occasionally sends NaN values in the compass
            compass = 0.0

        result = {
                'rgb': rgb,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                }

        if (self.backbone != 'latentTF'):
            lidar = input_data['lidar'][1][:, :3]
            result['lidar'] = lidar

        pos = self._get_position(result)
        result['gps'] = pos

        self.gps_buffer.append(pos)
        denoised_pos = np.average(self.gps_buffer, axis=0)
        self._denoised_pos = denoised_pos   # expose to run_step for recovery ego_pos

        waypoint_route = self._route_planner.run_step(denoised_pos)
        next_wp, next_cmd = waypoint_route[1] if len(waypoint_route) > 1 else waypoint_route[0]
        result['next_command'] = next_cmd.value

        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-denoised_pos[0], next_wp[1]-denoised_pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        return result

    @torch.inference_mode() # Faster version of torch_no_grad
    def run_step(self, input_data, timestamp):
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            self.control = control        

        # Need to run this every step for GPS denoising
        tick_data = self.tick(input_data)

        # repeat actions twice to ensure LiDAR data availability
        if self.step % self.config.action_repeat == 1:
            self.update_gps_buffer(self.control, tick_data['compass'], tick_data['speed'])
            return self.control

        # prepare image input
        image = self.prepare_image(tick_data)

        num_points = None
        if(self.backbone == 'latentTF'): # Image only method
            lidar_bev = torch.zeros((1, 2, self.config.lidar_resolution_width, self.config.lidar_resolution_height)).to('cuda', dtype=torch.float32) #Dummy data
        else:
            # prepare LiDAR input
            if (self.config.use_point_pillars == True):
                lidar_cloud = deepcopy(input_data['lidar'][1])
                lidar_cloud[:, 1] *= -1  # invert
                lidar_bev = [torch.tensor(lidar_cloud).to('cuda', dtype=torch.float32)]
                num_points = [torch.tensor(len(lidar_cloud)).to('cuda', dtype=torch.int32)]
            else:
                lidar_bev = self.prepare_lidar(tick_data)

        
        # prepare goal location input
        target_point_image, target_point = self.prepare_goal_location(tick_data)

        # prepare velocity input
        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32) # used by controller
        velocity = gt_velocity.reshape(1, 1) # used by transfuser

        # unblock
        is_stuck = False
        # divide by 2 because we process every second frame
        # 1100 = 55 seconds * 20 Frames per second, we move for 1.5 second = 30 frames to unblock
        # NOTE: when the RecoveryFSM is enabled it OWNS the is_stuck decision and
        # is computed AFTER obstacles are available (see the recovery block just
        # before the base controller). The original blind creep below is used
        # only as the fallback when use_recovery_fsm is False (baseline / ablation B).
        if not getattr(self, 'use_recovery_fsm', False):
            if(self.stuck_detector > self.config.stuck_threshold and self.forced_move < self.config.creep_duration):
                print("Detected agent being stuck. Move for frame: ", self.forced_move)
                is_stuck = True
                self.forced_move += 1


        # forward pass
        with torch.no_grad():
            pred_wps = []
            bounding_boxes = []
            for i in range(self.model_count):
                rotated_bb = []
                if (self.backbone == 'transFuser'):
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity,
                                                          num_points=num_points, save_path=SAVE_PATH, stuck_detector=self.stuck_detector,
                                                          forced_move=is_stuck, debug=self.config.debug, rgb_back=self.rgb_back)
                elif (self.backbone == 'late_fusion'):
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
                elif (self.backbone == 'geometric_fusion'):
                    bev_points = list()
                    cam_points = list()

                    curr_bev_points, curr_cam_points = lidar_bev_cam_correspondences(deepcopy(tick_data['lidar']), lidar_bev, image, self.step, False)
                    bev_points.append(torch.from_numpy(curr_bev_points).unsqueeze(0))
                    cam_points.append(torch.from_numpy(curr_cam_points).unsqueeze(0))

                    bev_points = bev_points[0].long().to('cuda', dtype=torch.int64)
                    cam_points = cam_points[0].long().to('cuda', dtype=torch.int64)
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, bev_points, cam_points, num_points=num_points)
                elif (self.backbone == 'latentTF'):
                    pred_wp, rotated_bb = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
                elif (self.backbone=='swin_ptt'):
                    pred_wp, rotated_bb = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
                
                else:
                    raise ("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")

                pred_wps.append(pred_wp)
                bounding_boxes.append(rotated_bb)

        bbs_vehicle_coordinate_system = self.non_maximum_suppression(bounding_boxes, self.iou_treshold_nms)

        self.bb_buffer.append(bbs_vehicle_coordinate_system)
        self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0) #Average the predictions from the ensembles

        # transform to local coordinates
        pred_wp_transformed = []
        for i, degree in enumerate(self.aug_degrees):
            rad = np.deg2rad(degree)
            degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                                [-np.sin(rad), np.cos(rad)]])
            # inverse
            degree_matrix = degree_matrix.T
            cur_pred_wp = self.pred_wp[i].detach().cpu().numpy()
            transformed_wp = (degree_matrix @ cur_pred_wp.T).T
            pred_wp_transformed.append(transformed_wp)

        self.pred_wp = np.stack(pred_wp_transformed, axis=0)
        self.pred_wp = torch.median(torch.from_numpy(self.pred_wp).to('cuda', dtype=torch.float32), dim=0, keepdims=True)[0]
        ###########
        if (self.backbone == 'latentTF'):
            safety_box = []
            if(self.bb_detected_in_front_of_vehicle(gt_velocity) == True):
                safety_box.append(True)
        else:
            # safety check
            safety_box = deepcopy(tick_data['lidar'])
            safety_box[:, 1] *= -1  # invert

            # z-axis
            safety_box      = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
            safety_box      = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

            # y-axis
            safety_box      = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
            safety_box      = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

            # x-axis
            safety_box      = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
            safety_box      = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]

        # ---- RECOVERY FSM: build obstacles once, decide is_stuck (policy-gap fix) ----
        # Obstacles are reused by the MPC block below (built once here).
        recovery_obstacles = []
        rec = None
        if (self.use_swift_mpc or self.use_recovery_fsm
                or getattr(self, 'use_waypoint_safety', False)
                or getattr(self, 'use_safety_head', False)
                or getattr(self, 'use_traj_check', False)):
            curr_bbs_rec = list(self.bb_buffer[-1]) if len(self.bb_buffer) > 0 else []
            if self.use_3frame_velocity:
                recovery_obstacles = self._obstacles_from_bbs_3frame(
                    curr_bbs_rec, self.prev_bbs_mpc, self.prev2_bbs_mpc,
                    ego_speed=float(gt_velocity), frame_dt=0.1)
            else:
                recovery_obstacles = self._obstacles_from_bbs(
                    curr_bbs_rec, self.prev_bbs_mpc,
                    ego_speed=float(gt_velocity), frame_dt=0.1)

        if self.use_recovery_fsm:
            cx_max = getattr(self.config, 'rec_obstacle_ahead_x', 12.0)
            cy_max = getattr(self.config, 'rec_obstacle_ahead_y', 2.5)
            v_move = getattr(self.config, 'rec_moving_speed', 0.5)
            # trigger distance: a deadlock requires a blocker genuinely CLOSE
            # ahead. A red-light / stop-line wait has OPEN road ahead (no nearby
            # detected vehicle), so requiring a close blocker prevents the FSM
            # from creeping through a red light (route-1 regression).
            trig_x = getattr(self.config, 'rec_trigger_x', 8.0)
            def _ahead(o):
                return 0.0 < o['center'][0] < cx_max and abs(o['center'][1]) < cy_max
            def _blocking(o):   # close enough to be a real deadlock blocker
                return 0.0 < o['center'][0] < trig_x and abs(o['center'][1]) < cy_max
            obstacle_ahead = any(_blocking(o) for o in recovery_obstacles)
            moving_obstacle_ahead = any(
                _ahead(o) and (o['vel'][0] ** 2 + o['vel'][1] ** 2) > v_move ** 2
                for o in recovery_obstacles)
            made_progress = float(gt_velocity) > getattr(self.config, 'rec_stall_speed', 0.1)

            # nearest dead-ahead obstacle for the safety gate
            _cands_ahead = [o for o in recovery_obstacles if o['center'][0] > 0]
            if _cands_ahead:
                _nrst = min(_cands_ahead, key=lambda o: o['center'][0])
                near_x, near_y = float(_nrst['center'][0]), float(_nrst['center'][1])
            else:
                near_x, near_y = None, None

            # PROXIMITY VETO: block creep if ANY obstacle (including ones off to
            # the SIDE, like a motorcycle the creep would turn into) is within a
            # close radius in the forward half-plane. The centered-obstacle gate
            # alone missed side obstacles -> the motorcycle collision. This
            # checks every obstacle, not just the nearest-centered one.
            _veto_r = getattr(self.config, 'rec_proximity_radius', 6.0)
            _veto_x = getattr(self.config, 'rec_proximity_x', 7.0)
            _veto_y = getattr(self.config, 'rec_proximity_y', 3.0)
            close_obstacle = False
            _close_desc = None
            for o in recovery_obstacles:
                ox, oy = o['center'][0], o['center'][1]
                if ox <= -1.0:            # behind us, ignore
                    continue
                dist = (ox * ox + oy * oy) ** 0.5
                # veto if within radius, OR inside a wide forward box that
                # covers the lane the creep+turn would sweep through
                if dist < _veto_r or (0.0 < ox < _veto_x and abs(oy) < _veto_y):
                    close_obstacle = True
                    _close_desc = 'x=%.1f,y=%.1f,d=%.1f' % (ox, oy, dist)
                    break

            # SAFETY GATE (Fix 2): is a forward creep safe here? Uses predicted
            # waypoint curvature + lateral deviation + obstacle centering.
            try:
                _wp_gate = self.pred_wp[0].detach().cpu().numpy()
            except Exception:
                _wp_gate = None
            creep_safe, creep_safe_reason = self._creep_safety_gate(
                _wp_gate, near_x, near_y,
                max_curve_rad=getattr(self.config, 'rec_max_curve_rad', 0.25),
                max_lat_dev=getattr(self.config, 'rec_max_lat_dev', 1.5),
                obstacle_center_y=getattr(self.config, 'rec_obstacle_center_y', 1.2),
                stop_distance=getattr(self.config, 'rec_stop_distance', 5.0))

            # proximity veto overrides everything: any close obstacle -> no creep
            if close_obstacle:
                creep_safe = False
                creep_safe_reason = 'proximity_' + (_close_desc or '')

            # ego world position for same-spot global give-up (Fix 1)
            try:
                ego_pos = (float(self._denoised_pos[0]), float(self._denoised_pos[1]))
            except Exception:
                ego_pos = None

            rec = self.recovery.update(float(gt_velocity), obstacle_ahead,
                                       moving_obstacle_ahead, made_progress,
                                       creep_safe=creep_safe, ego_pos=ego_pos)
            is_stuck = bool(rec['creep'])   # FSM now owns the creep decision
            if is_stuck:
                self.forced_move += 1       # keep counter live for downstream 'initial frame' logic
            else:
                self.forced_move = 0
            if self.step % 10 == 0:
                # DIAGNOSTIC: obstacles, nearest, and creep-safety verdict.
                _nob = len(recovery_obstacles)
                _near = f"x={near_x:.1f},y={near_y:.1f}" if near_x is not None else None
                print(f"[REC] step={self.step} state={rec['state']} reason={rec['reason']} "
                      f"is_stuck={is_stuck} obs_ahead={obstacle_ahead} "
                      f"moving={moving_obstacle_ahead} v={float(gt_velocity):.2f} "
                      f"nobs={_nob} nearest={_near} safe={creep_safe}({creep_safe_reason}) "
                      f"att={self.recovery.attempt_count}")

        # Base controller ALWAYS computes speed and default steering.
        steer, throttle, brake = self.nets[0].control_pid(self.pred_wp, gt_velocity, is_stuck)

        # ---- WAYPOINT SAFETY FILTER (design A: reject-and-brake) ----
        # Before the model's waypoints are allowed to drive the car, check that
        # following them does not run the ego INTO an obstacle. If it would,
        # override to a brake. This never steers/creeps -> cannot itself cause
        # a collision, lane departure, or red-light run.
        if getattr(self, 'use_waypoint_safety', False):
            try:
                _wp_safe = self.pred_wp[0].detach().cpu().numpy()
                unsafe, wp_reason, wp_info = self._waypoint_safety_check(
                    _wp_safe, recovery_obstacles, float(gt_velocity),
                    lidar_pos_x=float(self.config.lidar_pos[0]),
                    safety_margin=getattr(self.config, 'wp_safety_margin', 0.9))
                if unsafe:
                    print(
                        "[SAFETY-XAI]",
                        self._format_safety_attribution(unsafe, wp_reason, wp_info),
                        flush=True
                    )

                    # ========================================================
                    # SAVE REAL SAFETY EVENT FOR PAPER VISUALIZATION
                    # ========================================================
                    if (
                        getattr(self, 'save_safety_xai', False)
                        and
                        (self.step - self.safety_xai_last_save_step)
                        >= self.safety_xai_min_gap
                    ):
                        try:
                            def _xai_cpu_value(v):
                                """Convert nested tensors/arrays to CPU-safe values."""
                                if torch.is_tensor(v):
                                    return v.detach().cpu()
                                if isinstance(v, np.ndarray):
                                    return v.copy()
                                if isinstance(v, dict):
                                    return {k: _xai_cpu_value(val) for k, val in v.items()}
                                if isinstance(v, list):
                                    return [_xai_cpu_value(val) for val in v]
                                if isinstance(v, tuple):
                                    return tuple(_xai_cpu_value(val) for val in v)
                                return v

                            # RGB panorama exactly as consumed by the policy tick.
                            rgb_np = np.asarray(tick_data['rgb']).copy()

                            # Raw LiDAR point cloud from the same simulation step.
                            lidar_np = None
                            if 'lidar' in tick_data:
                                lidar_np = np.asarray(tick_data['lidar']).copy()

                            # Final policy waypoints passed to the shield.
                            wp_np = np.asarray(_wp_safe, dtype=np.float32).copy()

                            # All obstacle states used by the shield.
                            obs_save = _xai_cpu_value(deepcopy(recovery_obstacles))

                            # Safety attribution produced by the real shield.
                            info_save = _xai_cpu_value(deepcopy(wp_info))

                            # The attribution implementation records the
                            # geometry-based VRU/small-object flag under this key.
                            vru_like = bool(
                                wp_info.get(
                                    'is_small_or_vru_like',
                                    wp_info.get('vru_like', False)
                                )
                            )
                            event_type = 'vru' if vru_like else 'vehicle'

                            self.safety_xai_count += 1
                            filename = (
                                f"safety_{event_type}_"
                                f"step_{self.step:06d}_"
                                f"{self.safety_xai_count:04d}.pt"
                            )
                            save_file = os.path.join(
                                self.safety_xai_dir,
                                filename
                            )

                            event = {
                                'step': int(self.step),
                                'timestamp': float(timestamp),
                                'event_type': event_type,

                                'rgb': rgb_np,
                                'lidar': lidar_np,
                                'pred_waypoints': wp_np,
                                'obstacles': obs_save,

                                'ego_speed_mps': float(gt_velocity),
                                'unsafe': bool(unsafe),
                                'reason': str(wp_reason),
                                'wp_info': info_save,

                                # Useful navigation/context signals from the
                                # exact same frame. target_point may have been
                                # converted to tensors earlier in run_step.
                                'target_point': _xai_cpu_value(
                                    deepcopy(tick_data.get('target_point', None))
                                ),
                                'compass': float(tick_data.get('compass', 0.0)),
                                'next_command': tick_data.get('next_command', None),
                            }

                            torch.save(event, save_file)
                            self.safety_xai_last_save_step = self.step

                            print(
                                f"[SAFETY-XAI-SAVE] {save_file}",
                                flush=True
                            )

                        except Exception as save_e:
                            print(
                                "[SAFETY-XAI-SAVE] FAILED:",
                                repr(save_e),
                                flush=True
                            )
            except Exception as e:
                if self.step % 50 == 0:
                    print(f"[WP-SAFE] skipped: {repr(e)}")

        # ---- LEARNED SAFETY HEAD (detection + brake) ----
        # Trained detection head predicts P(collision) for the model's waypoints.
        # If P > threshold, brake -- BUT only when braking can actually prevent a
        # collision. Guards (added after the 24-route run showed the head
        # over-braked -> blocked/timeout on routes that had no collision):
        #   G1 threshold raised (config/env, default now 0.8).
        #   G2 don't brake when already slow: a car crawling (<1.5 m/s) that
        #      hard-stops is what produces "Agent got blocked". If we're slow,
        #      the base controller + safety box already handle close obstacles;
        #      the head's job is preventing FAST approaches into traffic.
        #   G3 require an obstacle genuinely CLOSE ahead: the hazard label fires
        #      on a 2s forecast, but braking only helps if something is near.
        if getattr(self, 'use_safety_head', False) and self.safety_head is not None:
            try:
                _wp_head = self.pred_wp[0].detach().cpu().numpy()
                p_coll = self.safety_head.collision_prob(_wp_head, recovery_obstacles)
                v_now = float(gt_velocity)
                # G3: is there a close obstacle roughly ahead?
                min_brake_speed = getattr(self.config, 'safety_head_min_speed', 1.5)
                close_ahead_x = getattr(self.config, 'safety_head_close_x', 12.0)
                close_ahead_y = getattr(self.config, 'safety_head_close_y', 3.0)
                close_ahead = any(0.0 < o['center'][0] < close_ahead_x
                                  and abs(o['center'][1]) < close_ahead_y
                                  for o in recovery_obstacles)

                # VRU sensitivity: motorcycles / cyclists have SMALL extents.
                # Colliding with them is the worst outcome and braking is more
                # likely to help when they are ahead. So if a small (VRU) obstacle
                # is close ahead, use a LOWER threshold -> brake more readily.
                vru_size = getattr(self.config, 'safety_head_vru_size', 1.0)   # max half-extent
                vru_x = getattr(self.config, 'safety_head_vru_x', 12.0)
                vru_y = getattr(self.config, 'safety_head_vru_y', 3.5)
                vru_close = any(
                    0.0 < o['center'][0] < vru_x
                    and abs(o['center'][1]) < vru_y
                    and max(o.get('extent', (2.4, 1.1))) < vru_size
                    for o in recovery_obstacles)
                # effective threshold: normal, but lower when a VRU is close
                eff_thresh = self.safety_head_thresh
                if vru_close:
                    eff_thresh = getattr(self.config, 'safety_head_vru_thresh', 0.4)

                do_brake = (p_coll > eff_thresh
                            and v_now > min_brake_speed      # G2: not already slow
                            and (close_ahead or vru_close))  # G3: real obstacle near
                if do_brake:
                    throttle = 0.0
                    brake = True
                if self.step % 10 == 0:
                    print(f"[HEAD] step={self.step} p_coll={p_coll:.2f} "
                          f"eff_thresh={eff_thresh:.2f}{'(VRU)' if vru_close else ''} "
                          f"{'BRAKE' if do_brake else 'ok'} "
                          f"close={close_ahead} vru={vru_close} "
                          f"nobs={len(recovery_obstacles)} v={v_now:.2f}")
            except Exception as e:
                if self.step % 50 == 0:
                    print(f"[HEAD] skipped: {repr(e)}")

        # ---- LEVEL 1: TRAJECTORY CHECK (other vehicles' predicted paths) ----
        # The head above is SNAPSHOT-based: it only sees where vehicles ARE, so it
        # reacts once they are already close -- too late for CROSSING traffic at
        # intersections (route 6 / route 9 failures). This forecasts each nearby
        # vehicle ~2s ahead and brakes EARLY if its predicted path will cross ours.
        # Runs independently of the head (TRAJ_CHECK=1); brake if EITHER fires.
        if getattr(self, 'use_traj_check', False) and self.traj_checker is not None:
            try:
                _wp_traj = self.pred_wp[0].detach().cpu().numpy()
                v_now = float(gt_velocity)
                hazard, tinfo = self.traj_checker.check(
                    _wp_traj, recovery_obstacles, v_now)
                # same guard as the head: never freeze an already-crawling car
                traj_brake = hazard and v_now > self.traj_min_speed
                if traj_brake:
                    throttle = 0.0
                    brake = True
                if self.step % 10 == 0:
                    ttc = tinfo.get('ttc_s')
                    print(f"[TRAJ] step={self.step} "
                          f"{'BRAKE' if traj_brake else 'ok'} "
                          f"hazard={hazard} "
                          f"ttc={ttc if ttc else '-'}s "
                          f"others={tinfo.get('n_others',0)} v={v_now:.2f}")
            except Exception as e:
                if self.step % 50 == 0:
                    print(f"[TRAJ] skipped: {repr(e)}")

        # SWIFT-MPC option B: steering assist only. Throttle/brake stay with
        # control_pid + the agent's safety box (baseline RC preserved). The
        # MPC overrides STEERING only when the model's own path is
        # geometrically blocked and a feasible avoidance maneuver exists.
        if self.use_swift_mpc:
            try:
                wp_np = self.pred_wp[0].detach().cpu().numpy()   # (4,2) local frame
                if getattr(self.config, 'mpc_axis_swap', 0):
                    wp_np = wp_np[:, ::-1].copy()
                yl = getattr(self.config, 'mpc_y_left', 1.0)
                if yl != 1.0:
                    wp_np = wp_np.copy(); wp_np[:, 1] *= yl
                curr_bbs = list(self.bb_buffer[-1]) if len(self.bb_buffer) > 0 else []
                # reuse the obstacle list already built for the FSM (built once)
                obstacles = recovery_obstacles
                if yl != 1.0:
                    for ob in obstacles:
                        ob['center'] = (ob['center'][0], ob['center'][1] * yl)
                        ob['vel'] = (ob['vel'][0], ob['vel'][1] * yl)

                # LOW-SPEED ESCAPE: when the FSM is in CREEP_STEER (allow_mpc_steer)
                # and we're ~stopped, let the MPC steer AROUND the blocker. The
                # stock steer_assist gates on speed and does nothing at v~=0.
                use_lowspeed = (self.use_mpc_lowspeed_escape and rec is not None
                                and rec.get('allow_mpc_steer', False)
                                and float(gt_velocity) < 1.0)
                if use_lowspeed:
                    assist = self.mpc.steer_assist_lowspeed(
                        wp_np, obstacles,
                        getattr(self.config, 'rec_creep_speed', 2.0))
                else:
                    assist = self.mpc.steer_assist(float(gt_velocity), wp_np, obstacles)
                if assist['override']:
                    steer = self.mpc.delta_to_steer(assist['delta'])
                if self.step % 10 == 0:
                    tag = "MPC-LS" if use_lowspeed else "MPC-B"
                    print(f"[{tag}] step={self.step} override={assist['override']} "
                          f"d={assist['delta']:+.2f} reason={assist['reason']} "
                          f"nobs={len(obstacles)} steer={steer:+.2f} "
                          f"thr={throttle:.2f} brk={brake:.2f}")
            except Exception as e:
                print(f"[MPC-B] EXCEPTION -> pure PID: {repr(e)}")

        # advance bb history for velocity estimation (prev2 <- prev <- curr).
        # Done once, unconditionally, so 3-frame velocity works even when the
        # MPC is disabled (FSM-only ablation).
        if self.use_swift_mpc or self.use_recovery_fsm:
            _curr_bbs_now = list(self.bb_buffer[-1]) if len(self.bb_buffer) > 0 else []
            self.prev2_bbs_mpc = self.prev_bbs_mpc
            self.prev_bbs_mpc = _curr_bbs_now

        if is_stuck and self.forced_move==1: # no steer for initial frame when unblocking
            steer = 0.0

        # steer modulation
        if brake or is_stuck:
            steer *= self.steer_damping
        if(gt_velocity < 0.1): # 0.1 is just an arbitrary low number to threshhold when the car is stopped
            self.stuck_detector += 1
        elif(gt_velocity > 0.1 and is_stuck == False):
            self.stuck_detector = 0
            self.forced_move    = 0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        # Safety controller. Stops the car in case something is directly in front of it.
        if self.use_lidar_safe_check:
            emergency_stop = (len(safety_box) > 0) #Checks if the List is empty
            # During a low-speed MPC ESCAPE the FSM is deliberately steering
            # AROUND a static blocker. The safety box is a straight-ahead strip,
            # so a hard stop here would veto the very swerve we intend. In that
            # case keep the MPC steering and creep slowly instead of full stop.
            escaping = (rec is not None and rec.get('allow_mpc_steer', False)
                        and self.use_mpc_lowspeed_escape)
            if ((emergency_stop == True) and (is_stuck == True)):  # We only use the saftey box when unblocking
                if escaping and abs(float(steer)) > 0.05:
                    # steering laterally to escape: allow a gentle creep, don't hard-stop
                    print("Escape maneuver: object ahead but steering around it. Step:", self.step)
                    control.steer = float(steer)
                    control.throttle = float(min(0.3, throttle if throttle else 0.3))
                    control.brake = float(False)
                else:
                    print("Detected object directly in front of the vehicle. Stopping. Step:", self.step)
                    control.steer = float(steer)
                    control.throttle = float(0.0)
                    control.brake = float(True)
                    # Will overwrite the stuck detector. If we are stuck in traffic we do want to wait it out.

        self.control = control

        self.update_gps_buffer(self.control, tick_data['compass'], tick_data['speed'])
        return control

    def bb_detected_in_front_of_vehicle(self, ego_speed):
        if (len(self.bb_buffer) < 1):  # We only start after we have 4 time steps.
            return False

        collision_predicted = False

        # These are the dimensions of the standard ego vehicle
        extent_x = self.config.ego_extent_x
        extent_y = self.config.ego_extent_y
        extent_z = self.config.ego_extent_z
        extent = carla.Vector3D(extent_x, extent_y, extent_z)

        # Safety box
        bremsweg = ((ego_speed.cpu().numpy().item() * 3.6) / 10.0) ** 2 / 2.0  # Bremsweg formula for emergency break
        safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0)  # plus one meter is the car.

        center_safety_box = carla.Location(x=safety_x, y=0.0, z=1.0)

        safety_bounding_box = carla.BoundingBox(center_safety_box, extent)
        safety_bounding_box.rotation = carla.Rotation(0.0,0.0,0.0)

        for bb in self.bb_buffer[-1]:
            bb_orientation = self.get_bb_yaw(bb)
            bb_extent_x = 0.5 * np.sqrt((bb[3, 0] - bb[0, 0]) ** 2 + (bb[3, 1] - bb[0, 1]) ** 2)
            bb_extent_y = 0.5 * np.sqrt((bb[0, 0] - bb[1, 0]) ** 2 + (bb[0, 1] - bb[1, 1]) ** 2)
            bb_extent_z = 1.0  # We just give them some arbitrary height. Does not matter
            loc_local = carla.Location(bb[4,0], bb[4,1], 0.0)
            extent_det = carla.Vector3D(bb_extent_x, bb_extent_y, bb_extent_z)
            bb_local = carla.BoundingBox(loc_local, extent_det)
            bb_local.rotation = carla.Rotation(0.0, np.rad2deg(bb_orientation).item(), 0.0)

            if (self.check_obb_intersection(safety_bounding_box, bb_local) == True):
                collision_predicted = True

        return collision_predicted

    def non_maximum_suppression(self, bounding_boxes, iou_treshhold):
        filtered_boxes = []
        bounding_boxes = np.array(list(itertools.chain.from_iterable(bounding_boxes)), dtype=np.object)

        if(bounding_boxes.size == 0): #If no bounding boxes are detected can't do NMS
            return filtered_boxes


        confidences_indices = np.argsort(bounding_boxes[:, 2])
        while (len(confidences_indices) > 0):
            idx = confidences_indices[-1]
            current_bb = bounding_boxes[idx, 0]
            filtered_boxes.append(current_bb)
            confidences_indices = confidences_indices[:-1] #Remove last element from the list

            if(len(confidences_indices) == 0):
                break

            for idx2 in deepcopy(confidences_indices):
                if(self.iou_bbs(current_bb, bounding_boxes[idx2, 0]) > iou_treshhold): # Remove BB from list
                    confidences_indices = confidences_indices[confidences_indices != idx2]

        return filtered_boxes

    def update_gps_buffer(self, control, theta, speed):
        yaw = np.array([(theta - np.pi/2.0)])
        speed = np.array([speed])
        action = np.array(np.stack([control.steer, control.throttle, control.brake], axis=-1))

        #Update gps locations
        for i in range(len(self.gps_buffer)):
            loc =self.gps_buffer[i]
            loc_temp = np.array([loc[1], -loc[0]]) #Bicycle model uses a different coordinate system
            next_loc_tmp, _, _ = self.ego_model.forward(loc_temp, yaw, speed, action)
            next_loc = np.array([-next_loc_tmp[1], next_loc_tmp[0]])
            self.gps_buffer[i] = next_loc

        return None

    def get_bb_yaw(self, box):
        location_2 = box[2]
        location_3 = box[3]
        location_4 = box[4]
        center_top = (0.5 * (location_3 - location_2)) + location_2
        vector_top = center_top - location_4
        rotation_yaw = np.arctan2(vector_top[1], vector_top[0])

        return rotation_yaw

    def prepare_image(self, tick_data):
        image = Image.fromarray(tick_data['rgb'])
        image_degrees = []
        for degree in self.aug_degrees:
            crop_shift = degree / 60 * self.config.img_width
            rgb = torch.from_numpy(self.shift_x_scale_crop(image, scale=self.config.scale, crop=self.config.img_resolution, crop_shift=crop_shift)).unsqueeze(0)
            image_degrees.append(rgb.to('cuda', dtype=torch.float32))
        image = torch.cat(image_degrees, dim=0)
        return image

    def iou_bbs(self, bb1, bb2):
        a = Polygon([(bb1[0,0], bb1[0,1]), (bb1[1,0], bb1[1,1]), (bb1[2,0], bb1[2,1]), (bb1[3,0], bb1[3,1])])
        b = Polygon([(bb2[0,0], bb2[0,1]), (bb2[1,0], bb2[1,1]), (bb2[2,0], bb2[2,1]), (bb2[3,0], bb2[3,1])])
        intersection_area = a.intersection(b).area
        union_area = a.union(b).area
        iou = intersection_area / union_area
        return iou
    
    
    def dot_product(self, vector1, vector2):
        return (vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z)

    def cross_product(self, vector1, vector2):
        return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y, y=vector1.z * vector2.x - vector1.x * vector2.z, z=vector1.x * vector2.y - vector1.y * vector2.x)

    def get_separating_plane(self, rPos, plane, obb1, obb2):
        ''' Checks if there is a seperating plane
        rPos Vec3
        plane Vec3
        obb1  Bounding Box
        obb2 Bounding Box
        '''
        return (abs(self.dot_product(rPos, plane)) > (abs(self.dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane)) +
                                                      abs(self.dot_product((obb1.rotation.get_right_vector()   * obb1.extent.y), plane)) +
                                                      abs(self.dot_product((obb1.rotation.get_up_vector()      * obb1.extent.z), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_right_vector()   * obb2.extent.y), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_up_vector()      * obb2.extent.z), plane)))
                )
    
    def check_obb_intersection(self, obb1, obb2):
        RPos = obb2.location - obb1.location
        return not(self.get_separating_plane(RPos, obb1.rotation.get_forward_vector(), obb1, obb2) or
                   self.get_separating_plane(RPos, obb1.rotation.get_right_vector(),   obb1, obb2) or
                   self.get_separating_plane(RPos, obb1.rotation.get_up_vector(),      obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_forward_vector(), obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_right_vector(),   obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_up_vector(),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_up_vector()),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_up_vector()),      obb1, obb2))



    def prepare_lidar(self, tick_data):
        lidar_transformed = deepcopy(tick_data['lidar']) 
        lidar_transformed[:, 1] *= -1  # invert
        lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed)).unsqueeze(0)
        lidar_transformed_degrees = [lidar_transformed.to('cuda', dtype=torch.float32)]
        lidar_bev = torch.cat(lidar_transformed_degrees[::-1], dim=1)
        return lidar_bev

    def prepare_goal_location(self, tick_data):
        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
                                            torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

        target_point_image_degrees = []
        target_point_degrees = []
        for degree in self.aug_degrees:
            rad = np.deg2rad(degree)
            degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                                [-np.sin(rad), np.cos(rad)]])

            current_target_point = (degree_matrix @ target_point[0].cpu().numpy().reshape(2, 1)).T

            target_point_image = draw_target_point(current_target_point[0])
            target_point_image = torch.from_numpy(target_point_image)[None].to('cuda', dtype=torch.float32)
            target_point_image_degrees.append(target_point_image)
            target_point_degrees.append(torch.from_numpy(current_target_point))

        target_point_image = torch.cat(target_point_image_degrees, dim=0)
        target_point = torch.cat(target_point_degrees, dim=0).to('cuda', dtype=torch.float32)

        return target_point_image, target_point

    def scale_crop(self, image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
        (width, height) = (image.width // scale, image.height // scale)
        if scale != 1:
            image = image.resize((width, height))
        if crop_x is None:
            crop_x = width
        if crop_y is None:
            crop_y = height
            
        image = np.asarray(image)
        cropped_image = image[start_y:start_y+crop_y, start_x:start_x+crop_x]
        return cropped_image

    def shift_x_scale_crop(self, image, scale, crop, crop_shift=0):
        crop_h, crop_w = crop
        (width, height) = (int(image.width // scale), int(image.height // scale))
        im_resized = image.resize((width, height))
        image = np.array(im_resized)
        start_y = height//2 - crop_h//2
        start_x = width//2 - crop_w//2
        
        # only shift in x direction
        start_x += int(crop_shift // scale)
        cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
        cropped_image = np.transpose(cropped_image, (2,0,1))
        return cropped_image

    def destroy(self):
        del self.nets

# Taken from LBC
class RoutePlanner(object):
    def __init__(self, min_distance, max_distance):
        self.saved_route = deque()
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False

        self.mean = np.array([0.0, 0.0]) # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route

    def save(self):
        self.saved_route = deepcopy(self.route)

    def load(self):
        self.route = self.saved_route
        self.is_last = False

# Taken from World on Rails
class EgoModel():
    def __init__(self, dt=1./4):
        self.dt = dt
        
        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        self.front_wb    = -0.090769015
        self.rear_wb     = 1.4178275

        self.steer_gain  = 0.36848336
        self.brake_accel = -4.952399
        self.throt_accel = 0.5633837

    def forward(self, locs, yaws, spds, acts):
        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        steer = acts[..., 0:1].item()
        throt = acts[..., 1:2].item()
        brake = acts[..., 2:3].astype(np.uint8)

        if (brake):
            accel = self.brake_accel
        else:
            accel = self.throt_accel * throt

        wheel = self.steer_gain * steer

        beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
        yaws = yaws.item()
        spds = spds.item()
        next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
        next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
        next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

        next_locs = np.array([next_locs_0, next_locs_1])
        next_yaws = np.array(next_yaws)
        next_spds = np.array(next_spds)

        return next_locs, next_yaws, next_spds
