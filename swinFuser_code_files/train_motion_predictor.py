#!/usr/bin/env python3
"""
train_motion_predictor.py — Level 1, Stage 2.

Train a motion predictor for OTHER vehicles:

    f(state, history) -> future trajectory (8 points, ~1.6 s ahead)

Inputs (from motion_dataset.npz, built by build_motion_dataset.py)
  state   : (N,4)   [speed, yaw_rate, ext_x, ext_y]  of the vehicle
  hist    : (N,4,2) its last 4 positions, in ITS OWN frame (x fwd, y left)
  future  : (N,8,2) where it actually went, same frame   <- the target

Everything is in the VEHICLE'S OWN frame, so the model is viewpoint-invariant:
one predictor works for any vehicle regardless of where the ego is. At inference
we transform its output back into the ego frame for the intersection check.

Loss
----
L1 on the trajectory (robust to outliers; standard for motion forecasting).
We report ADE (average displacement error) and FDE (final displacement error),
the standard metrics -- these tell us in METRES how wrong the prediction is,
which is what matters for a collision check.

A constant-velocity baseline is also reported. The learned model MUST beat it,
otherwise it isn't worth having (many "motion predictors" fail this test).

Usage
  python train_motion_predictor.py --data motion_dataset.npz --epochs 20 \
         --out motion_predictor.pt
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

HIST = 4
FUT = 8
DT = 0.5   # dataset timestep (VERIFIED on real data: 0.499 s/frame, 2 Hz)


class MotionPredictor(nn.Module):
    def __init__(self, hist=HIST, fut=FUT, hidden=128):
        super().__init__()
        in_dim = 4 + hist * 2            # state + flattened history
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, fut * 2))
        self.fut = fut

    def forward(self, state, hist):
        B = state.shape[0]
        x = torch.cat([state, hist.reshape(B, -1)], dim=-1)
        return self.net(x).reshape(B, self.fut, 2)


def const_velocity_baseline(state, hist, fut=FUT):
    """Predict by extrapolating the last observed velocity (numpy).
    hist is in the vehicle's own frame, so current pos is ~(0,0) and the
    heading is +x. Velocity from the last two history points."""
    v = (hist[:, -1] - hist[:, -2]) / DT          # (N,2) m/s
    steps = np.arange(1, fut + 1)[None, :, None] * DT   # (1,F,1)
    return hist[:, -1][:, None, :] + v[:, None, :] * steps


def metrics(pred, gt):
    """ADE (mean over all steps) and FDE (final step), in metres."""
    d = np.linalg.norm(pred - gt, axis=-1)        # (N,F)
    return float(d.mean()), float(d[:, -1].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='motion_dataset.npz')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--out', default='motion_predictor.pt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fut', type=int, default=FUT,
                    help='prediction horizon in FRAMES (dt=0.5s each). '
                         '4 = 2.0s (recommended for collision checking), '
                         '8 = 4.0s (long horizon, larger error).')
    args = ap.parse_args()
    fut = min(args.fut, FUT)
    print(f"horizon: {fut} frames x {DT}s = {fut*DT:.1f} s ahead")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {dev}")

    d = np.load(args.data)
    S = d['state'].astype(np.float32)
    H = d['hist'].astype(np.float32)
    F = d['future'].astype(np.float32)[:, :fut]   # truncate to chosen horizon
    S = np.nan_to_num(S); H = np.nan_to_num(H); F = np.nan_to_num(F)
    print(f"data: {len(S)} vehicle samples")

    # --- constant-velocity baseline on the SAME val split (computed later) ---
    ds = TensorDataset(torch.from_numpy(S), torch.from_numpy(H), torch.from_numpy(F))
    n_val = int(len(ds) * args.val_frac); n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val],
                          generator=torch.Generator().manual_seed(args.seed))
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=2)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=2)

    # baseline over the val split
    vi = np.array(va.indices)
    b_pred = const_velocity_baseline(S[vi], H[vi], fut=fut)
    b_ade, b_fde = metrics(b_pred, F[vi])
    print(f"constant-velocity baseline:  ADE={b_ade:.2f} m  FDE={b_fde:.2f} m")
    print("  (the learned model must BEAT this, or it isn't worth having)\n")

    net = MotionPredictor(fut=fut).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    lossf = nn.L1Loss()

    def evaluate():
        net.eval()
        P, G = [], []
        with torch.no_grad():
            for s, h, f in vl:
                p = net(s.to(dev), h.to(dev)).cpu().numpy()
                P.append(p); G.append(f.numpy())
        return metrics(np.concatenate(P), np.concatenate(G))

    best = 1e9
    for ep in range(1, args.epochs + 1):
        net.train(); tot = 0.0
        for s, h, f in tl:
            s, h, f = s.to(dev), h.to(dev), f.to(dev)
            opt.zero_grad()
            loss = lossf(net(s, h), f)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(s)
        ade, fde = evaluate()
        print(f"ep{ep:02d} loss={tot/n_tr:.4f}  val ADE={ade:.2f} m  FDE={fde:.2f} m")
        if fde < best:
            best = fde
            torch.save({'model': net.state_dict(), 'ade': ade, 'fde': fde,
                        'dt': DT, 'hist': HIST, 'fut': fut}, args.out)

    print(f"\nBest val FDE = {best:.2f} m  (baseline FDE = {b_fde:.2f} m)")
    if best < b_fde:
        print(f"  -> the learned predictor BEATS constant-velocity by "
              f"{b_fde - best:.2f} m at the {fut*DT:.1f}s horizon.")
        if best < 3.0:
            print("  -> FDE < 3 m: accurate enough for a collision check. GOOD.")
        else:
            print(f"  -> WARNING: FDE {best:.1f} m is large (a car is ~4.5 m long).\n"
                  "     A collision check on this will be noisy. Try a shorter --fut.")
    else:
        print("  -> WARNING: it does NOT beat constant velocity. Do not use it; "
              "just extrapolate velocity instead (cheaper and as good).")
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()
