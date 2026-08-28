"""
Extensions to SamplingMPC for recovery, kept separate so each is an
independent ablation:

  Fix 2 : steer_assist_lowspeed(...)  -> lets the MPC propose a lateral escape
          while the ego is ~stopped. The stock steer_assist() gates on
          `sustain = V[:,-1] > creep_speed`, which is empty at v0~=0, so the
          MPC never engages in a deadlock. This branch evaluates the lattice
          at a small assumed creep speed instead, using the geometric hard-box
          feasibility only (no speed-scaled potential), and returns the
          feasible steering that makes the MOST forward progress.

  Fix 3 : obstacles_from_bbs_3frame(...) -> replaces 2-frame differencing with
          a 3-frame central finite difference for a smoother, lower-noise
          relative-velocity estimate. Better velocity -> better forecast in
          _potential -> earlier braking on genuine closing traffic (the
          reactive rear-end / junction-merge collisions, cluster 2).

Both are free functions / mixin-style methods so you can monkey-patch or copy
them onto your SamplingMPC instance without touching the reviewed core file.
"""
import os
import numpy as np


# --------------------------------------------------------------------------- #
# Fix 2: low-speed lateral escape
# --------------------------------------------------------------------------- #
def steer_assist_lowspeed(mpc, waypoints, obstacles, creep_v=2.0):
    """MPC steering proposal for the deadlocked (near-stopped) case.

    Rolls the candidate lattice out at an assumed creep speed `creep_v`, keeps
    only candidates that are geometrically collision-free (hard box) AND stay
    within the lane bound, and picks the one whose trajectory advances the
    ego FURTHEST forward (max terminal x). Ties broken toward smaller steering
    (smoothness).

    Returns dict: override(bool), delta(rad), reason, n_feasible.
    """
    if not obstacles:
        return {'override': False, 'delta': 0.0,
                'reason': 'ls_no_obstacles', 'n_feasible': -1}

    # Never swerve around a VRU in-corridor; braking/holding is the only
    # acceptable response (preserved from the main steer_assist RULE 1).
    vru_ahead = any(o.get('small', False)
                    and 0.0 < o['center'][0] < 14.0
                    and abs(o['center'][1]) < 2.2 for o in obstacles)
    if vru_ahead:
        return {'override': False, 'delta': 0.0,
                'reason': 'ls_vru_no_swerve', 'n_feasible': -1}

    # Roll out the lattice at the creep speed.
    P, V = mpc._rollout(creep_v)               # (C,N,2), (C,N)
    mpc._Vref_for_collide = V                  # geometric collision uses this
    _, collide = mpc._potential(P, obstacles)

    # dynamic feasibility (lateral accel) at creep speed is trivially fine, but
    # keep the check for consistency.
    a_lat = (V ** 2) * (np.abs(np.tan(mpc._delta_kn)) / mpc.L)
    dyn_infeasible = (a_lat > mpc.a_lat_max).any(axis=1)

    # lane bound: don't escape into the oncoming/adjacent lane by more than the
    # soft bound (guards against unseen static structure off-path).
    lat = np.abs(P[:, :, 1])                    # |y| deviation, path is ~+x here
    lane_ok = (lat.max(axis=1) < (mpc.lane_bound + 1.0))

    feasible = (~collide) & (~dyn_infeasible) & lane_ok
    if not feasible.any():
        # everything blocked: hold straight (caller keeps braking / waiting)
        return {'override': True, 'delta': 0.0,
                'reason': 'ls_all_blocked_hold', 'n_feasible': 0}

    forward = P[:, -1, 0]                        # terminal forward progress
    steer_mag = np.abs(mpc._delta)
    # score: maximize progress, mild penalty on steering magnitude
    score = forward - 0.3 * steer_mag
    score = np.where(feasible, score, -np.inf)
    i = int(np.argmax(score))

    delta = float(mpc._delta[i])
    # If the best feasible escape is essentially straight, there's no *lateral*
    # value to add here; report override with delta~0 so the caller still
    # creeps forward (progress) but doesn't crank the wheel.
    return {'override': True, 'delta': delta,
            'reason': 'ls_escape', 'n_feasible': int(feasible.sum())}


# --------------------------------------------------------------------------- #
# Fix 3: 3-frame velocity estimation
# --------------------------------------------------------------------------- #
def obstacles_from_bbs_3frame(curr_bbs, prev_bbs, prev2_bbs, ego_speed=0.0,
                              frame_dt=0.1, vru_width=0.6, vru_max_len=1.5, match_dist=5.0):
    """Like obstacles_from_bbs, but estimates each obstacle's velocity from a
    3-frame central difference (curr, prev, prev2) instead of a 2-frame
    backward difference. This halves the variance of the velocity estimate and
    removes the one-frame lag, which sharpens the forecast used by _potential
    and reduces late-braking rear-end / junction collisions.

    Matching is nearest-neighbour across consecutive frames. If prev2 is
    unavailable (start of episode / lost track) it falls back to the 2-frame
    estimate so behaviour degrades gracefully.

    Frame convention identical to obstacles_from_bbs: bb[4]=center (x fwd,
    y left); ego-frame relative motion is compensated by +ego_speed on x.
    """
    def _match(cx, cy, pool):
        if pool is None or len(pool) == 0:
            return None
        best, bd = None, match_dist ** 2
        for pb in pool:
            d = (float(pb[4, 0]) - cx) ** 2 + (float(pb[4, 1]) - cy) ** 2
            if d < bd:
                bd, best = d, pb
        return best

    obstacles = []
    for bb in curr_bbs:
        cx, cy = float(bb[4, 0]), float(bb[4, 1])
        ex = 0.5 * np.sqrt((bb[3, 0] - bb[0, 0]) ** 2 + (bb[3, 1] - bb[0, 1]) ** 2)
        ey = 0.5 * np.sqrt((bb[0, 0] - bb[1, 0]) ** 2 + (bb[0, 1] - bb[1, 1]) ** 2)

        vx = vy = 0.0
        p1 = _match(cx, cy, prev_bbs)
        if p1 is not None:
            p1x, p1y = float(p1[4, 0]), float(p1[4, 1])
            p2 = _match(p1x, p1y, prev2_bbs)
            if p2 is not None:
                # 3-frame central difference: v ~= (x_t - x_{t-2}) / (2 dt)
                p2x, p2y = float(p2[4, 0]), float(p2[4, 1])
                rvx = (cx - p2x) / (2.0 * frame_dt)
                rvy = (cy - p2y) / (2.0 * frame_dt)
            else:
                # fall back to 2-frame backward difference
                rvx = (cx - p1x) / frame_dt
                rvy = (cy - p1y) / frame_dt
            # relative -> absolute in the current ego frame
            vx = rvx + float(ego_speed)
            vy = rvy
            spd = np.hypot(vx, vy)
            # PER-AXIS corruption defense. The old combined test
            #   if spd > 8.0 or spd < 1.5: vx = vy = 0.0
            # zeroed a slow-merging car's REAL lateral velocity, so a car
            # changing into our lane read as STATIC. Its forecast then froze
            # in its origin lane, straight_blocked stayed False, the MPC
            # returned path_clear, and (option B never braking) the ego drove
            # into it -> the route-30/32 lane-change collisions.
            #   * still drop the whole vector if implausibly fast (ID switch)
            #   * else zero each axis under its OWN floor, with a much lower
            #     lateral floor so slow merges survive into the forecast.
            _zeroed = 'no'
            if spd > 8.0:
                vx = vy = 0.0
                _zeroed = 'fast'
            else:
                _lon_floor = 1.5   # longitudinal noise floor: unchanged
                _lat_floor = 0.5   # lateral floor: keep slow merges (KEY FIX)
                _zx = abs(vx) < _lon_floor
                _zy = abs(vy) < _lat_floor
                if _zx:
                    vx = 0.0
                if _zy:
                    vy = 0.0
                _zeroed = ('both' if (_zx and _zy) else
                           'lon' if _zx else 'lat' if _zy else 'no')
            if os.environ.get('OBS_DEBUG', '0') == '1':
                print(f"[OBS3] cx={cx:+.1f} cy={cy:+.1f} "
                      f"rvx={rvx:+.2f} rvy={rvy:+.2f} "
                      f"vx={vx:+.2f} vy={vy:+.2f} spd={spd:.2f} "
                      f"zeroed={_zeroed}")

        yaw = float(np.arctan2(bb[3, 1] - bb[0, 1], bb[3, 0] - bb[0, 0]))
        if cx < -3.0 or abs(cy) > 8.0 or cx > 32.0:
            continue
        obstacles.append({'center': (cx, cy), 'vel': (vx, vy),
                          'extent': (float(ex), float(ey)), 'yaw': yaw,
                          # VRU test uses the NARROW dimension, not the largest.
                          # A motorcycle is ~2.3 m LONG (half-extent 1.15) but
                          # only ~0.8 m wide (0.4), so max(ex,ey)=1.15 > 1.0
                          # classified it as a CAR and it lost every VRU
                          # protection (no-swerve rule, wider lateral margin,
                          # wider proximity veto). What makes a road user
                          # vulnerable is being NARROW, not being short.
                          'small': (min(ex, ey) < vru_width
                                    or max(ex, ey) < vru_max_len)})
    return obstacles
