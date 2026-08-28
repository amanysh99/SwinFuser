"""
train_detection_head.py -- trains the Waypoint Safety DETECTION head that
safety_head_infer.py loads at inference time.

TWO-PHASE ("general then specialize", the option ج we chose):
  Phase 1: train on the FULL dataset (all collision types) -- gives the head
           broad exposure to many near-crash situations.
  Phase 2: continue training (fine-tune) on the SIDESWIPE-only dataset -- the
           same-lane lateral-contact pattern behind Longest6 routes 30/32/33.
Phase 2 starts from Phase 1's weights, so it keeps the general knowledge and
sharpens on the target failure mode.

The architecture and normalization are IMPORTED from safety_head_infer.py so
they can never drift out of sync with inference (that file states it "MUST
mirror train_detection_head.py exactly" -- importing makes that automatic
rather than a promise).

CLASS IMBALANCE: the positive rate is ~4% (all) / ~1.5% (sideswipe). Trained
naively, the head learns to always predict "no collision" and looks accurate
while being useless. This script handles it with pos_weight in
BCEWithLogitsLoss (weights positive errors up by neg/pos) -- adjustable via
--pos_weight_scale.

GROUP-AWARE SPLIT: frames from one scenario are consecutive timesteps of the
same event and are highly correlated. A random frame-level split leaks
between train and val. This script splits by scenario_id (whole scenarios go
to train or val, never both) using the scenario_id array that
build_training_dataset.py now saves. Falls back to a warned frame-level split
if scenario_id is absent (older .npz).

Usage:
  # phase 1 only:
  python train_detection_head.py --data training_data_all.npz --out detection_head_p1.pt --epochs 40

  # phase 2 (fine-tune p1 on sideswipe):
  python train_detection_head.py --data training_data_sideswipe.npz \
      --init detection_head_p1.pt --out detection_head_final.pt \
      --epochs 30 --lr 3e-4

  # or run BOTH phases in one call:
  python train_detection_head.py --two_phase \
      --data training_data_all.npz --data2 training_data_sideswipe.npz \
      --out detection_head_final.pt
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn

# Import the EXACT architecture + normalization the agent uses at inference.
from safety_head_infer import DetectionHead, _normalize, MAX_OBS, N_WP


def load_npz(path):
    d = np.load(path)
    obs = d['obs'].astype(np.float32)
    wp = d['wp'].astype(np.float32)
    label = d['label'].astype(np.float32)
    sid = d['scenario_id'].astype(np.int64) if 'scenario_id' in d else None
    return obs, wp, label, sid


def normalize_batch(obs, wp):
    """Apply the shared _normalize (which is per-sample) across a whole array."""
    out_obs = np.empty_like(obs)
    out_wp = np.empty_like(wp)
    for i in range(len(obs)):
        out_obs[i], out_wp[i] = _normalize(obs[i], wp[i])
    return out_obs, out_wp


def group_split(sid, label, val_frac, seed):
    """Split by scenario so no scenario appears in both train and val.
    Tries to keep a comparable positive rate in val by splitting positive-
    containing and negative-only scenarios separately."""
    rng = np.random.RandomState(seed)
    scenarios = np.unique(sid)
    # which scenarios contain at least one positive frame
    pos_scen = np.array([s for s in scenarios if label[sid == s].max() > 0])
    neg_scen = np.array([s for s in scenarios if label[sid == s].max() == 0])
    rng.shuffle(pos_scen)
    rng.shuffle(neg_scen)

    def take(arr):
        k = max(1, int(round(len(arr) * val_frac))) if len(arr) else 0
        return set(arr[:k].tolist())

    val_scen = take(pos_scen) | take(neg_scen)
    val_mask = np.array([s in val_scen for s in sid])
    return ~val_mask, val_mask, len(pos_scen), len(val_scen)


def make_loader(obs, wp, label, batch, shuffle, device):
    obs_t = torch.from_numpy(obs)
    wp_t = torch.from_numpy(wp)
    y_t = torch.from_numpy(label)
    ds = torch.utils.data.TensorDataset(obs_t, wp_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    ys, ps = [], []
    for obs, wp, y in loader:
        obs, wp = obs.to(device), wp.to(device)
        logit = net(obs, wp)
        ps.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    # threshold-free-ish summary + a fixed 0.5 operating point
    pred = (p >= 0.5).astype(np.float32)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # a crude AUC via rank statistic (no sklearn dependency)
    auc = roc_auc(y, p)
    return dict(precision=prec, recall=rec, f1=f1, auc=auc,
                tp=tp, fp=fp, fn=fn, tn=tn, n_pos=int(y.sum()), n=len(y))


def roc_auc(y, p):
    """AUC via the Mann-Whitney U statistic; no external deps."""
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    avg = sum_ranks / counts
    ranks = avg[inv]
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def train_one_phase(net, tr_loader, va_loader, epochs, lr, pos_weight, device,
                     tag, weight_decay):
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_f1, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        net.train()
        tot = 0.0
        for obs, wp, y in tr_loader:
            obs, wp, y = obs.to(device), wp.to(device), y.to(device)
            opt.zero_grad()
            logit = net(obs, wp)
            loss = crit(logit, y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
        m = evaluate(net, va_loader, device)
        print(f"[{tag}] ep{ep:3d} loss={tot/max(len(tr_loader.dataset),1):.4f} "
              f"val: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} AUC={m['auc']:.3f} "
              f"(tp={m['tp']:.0f} fp={m['fp']:.0f} fn={m['fn']:.0f})")
        if m['f1'] >= best_f1:
            best_f1 = m['f1']
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_f1


def run_phase(data_path, net, args, device, tag):
    obs, wp, label, sid = load_npz(data_path)
    print(f"\n=== {tag}: {data_path} ===")
    print(f"  samples={len(label)}  positive={int(label.sum())}  "
          f"negative={int((label==0).sum())}")
    obs, wp = normalize_batch(obs, wp)

    if sid is not None:
        tr_mask, va_mask, n_pos_scen, n_val_scen = group_split(
            sid, label, args.val_frac, args.seed)
        print(f"  group split by scenario: {n_pos_scen} positive-containing "
              f"scenarios, {n_val_scen} scenarios held out for val")
    else:
        print("  WARNING: no scenario_id in this .npz -- falling back to a "
              "frame-level split, which LEAKS correlated frames between "
              "train/val and will overstate val metrics. Rebuild the dataset "
              "with the updated build_training_dataset.py to fix.")
        rng = np.random.RandomState(args.seed)
        va_mask = rng.rand(len(label)) < args.val_frac
        tr_mask = ~va_mask

    n_pos_tr = int(label[tr_mask].sum())
    n_neg_tr = int((label[tr_mask] == 0).sum())
    pos_weight = (n_neg_tr / max(n_pos_tr, 1)) * args.pos_weight_scale
    print(f"  train pos={n_pos_tr} neg={n_neg_tr} -> pos_weight={pos_weight:.1f} "
          f"(neg/pos x {args.pos_weight_scale})")
    if int(label[va_mask].sum()) == 0:
        print("  WARNING: val split has ZERO positive frames -- val recall/F1 "
              "will be meaningless. Increase --val_frac or check scenario "
              "balance.")

    tr = make_loader(obs[tr_mask], wp[tr_mask], label[tr_mask],
                     args.batch, True, device)
    va = make_loader(obs[va_mask], wp[va_mask], label[va_mask],
                     args.batch, False, device)
    net, best_f1 = train_one_phase(
        net, tr, va, args.epochs if tag == 'phase1' else args.epochs2,
        args.lr if tag == 'phase1' else args.lr2,
        pos_weight, device, tag, args.weight_decay)
    print(f"  {tag} best val F1 = {best_f1:.3f}")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='phase-1 dataset .npz')
    ap.add_argument('--data2', default=None,
                     help='phase-2 (sideswipe) dataset .npz; required with --two_phase')
    ap.add_argument('--two_phase', action='store_true',
                     help='run phase 1 on --data then phase 2 on --data2')
    ap.add_argument('--init', default=None,
                     help='checkpoint to initialise from (e.g. existing detection_head.pt '
                          'for fine-tuning, or a phase-1 output)')
    ap.add_argument('--out', required=True, help='output checkpoint path')
    ap.add_argument('--epochs', type=int, default=40, help='phase-1 epochs')
    ap.add_argument('--epochs2', type=int, default=30, help='phase-2 epochs')
    ap.add_argument('--lr', type=float, default=1e-3, help='phase-1 lr')
    ap.add_argument('--lr2', type=float, default=3e-4,
                     help='phase-2 lr (lower: fine-tuning, do not wreck phase-1 weights)')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--pos_weight_scale', type=float, default=1.0,
                     help='multiply the neg/pos pos_weight by this. >1 chases '
                          'recall (fewer missed collisions, more false brakes); '
                          '<1 the reverse.')
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net = DetectionHead().to(device)
    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location=device)
        state = ckpt['model'] if 'model' in ckpt else ckpt
        net.load_state_dict(state)
        print(f"initialised from {args.init}")

    if args.two_phase:
        if not args.data2:
            raise SystemExit("--two_phase requires --data2 (the sideswipe .npz)")
        net = run_phase(args.data, net, args, device, 'phase1')
        net = run_phase(args.data2, net, args, device, 'phase2')
    else:
        # single phase: tag as phase1 so it uses --epochs/--lr
        net = run_phase(args.data, net, args, device, 'phase1')

    torch.save({'model': net.state_dict()}, args.out)
    print(f"\nSaved: {args.out}")
    print("Load it in the agent via SafetyHead('%s'). Architecture matches "
          "safety_head_infer.py by import, so no shape mismatch is possible."
          % os.path.basename(args.out))


if __name__ == '__main__':
    main()
