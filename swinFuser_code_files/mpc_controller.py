"""
SWIFT-MPC: sampling-based model-predictive safety controller with
spatio-temporal potential fields, for use on top of an imitation planner
(SwinFuser ensemble waypoints + detected bounding boxes).

Key ideas
---------
- A deterministic lattice of candidate control sequences (constant steering
  angle x constant acceleration over a short horizon) is rolled out through a
  kinematic bicycle model.
- Each candidate trajectory is scored by: reference tracking (the model's
  predicted waypoints), a spatio-temporal potential field around detected
  vehicles (their positions FORECAST over the horizon from measured motion),
  bounded lateral deviation (guard against unseen static obstacles like
  fences), progress reward capped at the model's own target speed (so it never
  creeps through red lights), and control smoothness.
- Candidates whose peak potential exceeds a collision threshold are marked
  infeasible. If ALL candidates are infeasible, the controller commands a hard
  brake. This is the "check clearance over the WHOLE trajectory before
  committing" property that a per-frame reactive potential field lacks.
- Fully deterministic: same inputs -> same output (reproducibility).

Frame convention (standalone module)
------------------------------------
+x forward, +y LEFT, yaw counter-clockwise, angles in radians.
Positive steering angle (delta) turns LEFT (+y).
The mapping to CARLA's steer sign is done at integration time via
`steer_sign` (CARLA: steer>0 is right). VERIFY empirically at integration.

No dependencies beyond numpy.
"""
import numpy as np


class SamplingMPC:
    def __init__(self, config=None):
        g = (lambda k, d: getattr(config, k, d)) if config is not None else (lambda k, d: d)

        # Horizon / model
        self.N = int(g('mpc_horizon', 12))           # steps (12*0.25 = 3.0s)
        self.dt = float(g('mpc_dt', 0.25))           # s per step (horizon = N*dt)
        self.wp_dt = float(g('mpc_wp_dt', 0.5))      # s between model waypoints
        self.L = float(g('mpc_wheelbase', 2.9))      # m
        self.v_max = float(g('mpc_v_max', 12.0))     # m/s cap
        self.ego_halfwidth = float(g('mpc_ego_halfwidth', 1.0))
        self.ego_halflength = float(g('mpc_ego_halflength', 2.4))  # BUGFIX: ego is ~4.8m long, not a point

        # Candidate lattice
        self.n_steer = int(g('mpc_n_steer', 15))
        self.delta_max = float(g('mpc_delta_max', 0.70))   # rad at the wheels
        self.a_lat_max = float(g('mpc_a_lat_max', 4.5))    # m/s^2 lateral-accel cap
        self.s_phase = int(g('mpc_s_phase', 5))            # steps of out-steer before counter-steer
        self.return_frac = float(g('mpc_return_frac', 0.6))  # counter-steer = -frac * delta
        self.a_min = float(g('mpc_a_min', -4.0))
        self.a_max = float(g('mpc_a_max', 2.0))
        self.acc_levels = np.array(g('mpc_acc_levels', (-4.0, -2.0, -0.5, 0.75, 2.0)), float)

        # Cost weights
        self.w_track = float(g('mpc_w_track', 0.5))
        self.w_lane = float(g('mpc_w_lane', 4.0))
        self.w_pf = float(g('mpc_w_pf', 1.0))
        self.w_prog = float(g('mpc_w_prog', 3.0))
        self.w_speed = float(g('mpc_w_speed', 0.4))
        self.underspeed_frac = float(g('mpc_underspeed_frac', 0.15))  # driving slower than the model's target is cheap; faster is not
        self.speed_deadband = float(g('mpc_speed_deadband', 1.0))    # m/s: ignore overspeed within this band (no micro-braking)
        self.w_smooth = float(g('mpc_w_smooth', 8.0))
        self.lane_bound = float(g('mpc_lane_bound', 1.9))  # m soft bound on lateral deviation

        # Potential field
        self.pf_A = float(g('mpc_pf_gain', 60.0))
        self.pf_margin_lat = float(g('mpc_pf_margin_lat', 0.5))
        self.pf_margin_long = float(g('mpc_pf_margin_long', 1.0))
        self.vru_size = float(g('mpc_vru_size', 1.0))      # half-extent below which object = VRU
        self.vru_gain = float(g('mpc_vru_gain', 4.0))      # bikes/pedestrians repel much harder
        self.vru_margin = float(g('mpc_vru_margin', 1.0))
        self.hard_margin = float(g('mpc_hard_margin', 0.25))  # m: extra hard-box margin
        self.ego_front = float(g('mpc_ego_front', 1.0))       # m: ego bumper reach for hard-box (not full half-length)
        self.creep_speed = float(g('mpc_creep_speed', 1.0))   # m/s: below this, touching the hard box isn't a 'collision' (allows creeping to a stop)
        self.vel_inflate = float(g('mpc_vel_inflate', 0.5))   # s: inflate sigma along obstacle motion by v*this
        self.pf_vscale = float(g('mpc_pf_vscale', 5.0))      # m/s: PF scales with ego speed (risk ~ speed); floor below
        self.pf_vfloor = float(g('mpc_pf_vfloor', 0.1))      # min PF factor at standstill
        self.static_pf_frac = float(g('mpc_static_pf_frac', 0.15))  # soft-field fraction for STATIC obstacles
        self.side_clear_lat = float(g('mpc_side_clear_lat', 3.0))   # m: static obstacle beyond this lateral offset = another lane, no soft field
        self.w_term_blocked = float(g('mpc_w_term_blocked', 60.0))  # terminal cost: ending slow with a static
        self.term_lookahead = float(g('mpc_term_lookahead', 8.0))   # obstacle dead-ahead = a trapped state
        # rationale: a static obstacle's future is certain, so the geometric
        # hard box fully handles it; the soft field's job is anticipatory
        # caution around MOVING agents whose future is uncertain.

        # CARLA mapping
        self.steer_sign = float(g('mpc_steer_sign', 1.0))  # confirmed on-road: flipped from -1.0 after car steered into the boundary
        self.max_throttle = float(g('mpc_max_throttle', 0.75))

        # State
        self.prev_first_delta = 0.0

        # Precompute lattice
        deltas = np.linspace(-self.delta_max, self.delta_max, self.n_steer)
        D, A = np.meshgrid(deltas, self.acc_levels, indexing='ij')
        self._delta = D.ravel()          # first-step steering of each candidate
        self._acc = A.ravel()
        # Two-phase S-curve schedule: steer out for s_phase steps, then
        # counter-steer to straighten alongside the reference. A candidate can
        # therefore end PARALLEL at a lateral offset (e.g. beside a stopped
        # car) while keeping speed -- which is what makes overtaking visible
        # to the cost within one horizon.
        sched = np.ones(self.N)
        sched[self.s_phase:] = -self.return_frac
        self._delta_kn = self._delta[:, None] * sched[None, :]   # (C,N)

    # ------------------------------------------------------------------ #
    def _rollout(self, v0):
        """Vectorized kinematic-bicycle rollout of all candidates."""
        C = self._delta.shape[0]
        x = np.zeros(C); y = np.zeros(C); th = np.zeros(C)
        v = np.full(C, max(0.0, float(v0)))
        P = np.zeros((C, self.N, 2)); V = np.zeros((C, self.N))
        for k in range(self.N):
            x = x + v * np.cos(th) * self.dt
            y = y + v * np.sin(th) * self.dt
            th = th + (v / self.L) * np.tan(self._delta_kn[:, k]) * self.dt
            v = np.clip(v + self._acc * self.dt, 0.0, self.v_max)
            P[:, k, 0] = x; P[:, k, 1] = y; V[:, k] = v
        return P, V

    # ------------------------------------------------------------------ #
    def _reference(self, waypoints):
        """Interpolate/extrapolate the model's waypoints onto horizon times.
        Returns ref positions (N,2) and per-step reference speed (N,)."""
        wp = np.asarray(waypoints, float).reshape(-1, 2)
        if not np.isfinite(wp).all():
            wp = np.nan_to_num(wp, nan=0.0, posinf=0.0, neginf=0.0)
        if wp.shape[0] == 0:
            return np.zeros((self.N, 2)), np.zeros(self.N)
        tw = (np.arange(wp.shape[0]) + 1) * self.wp_dt
        tk = (np.arange(self.N) + 1) * self.dt

        # Desired speed from the model's waypoint spacing. Use the MEDIAN of
        # consecutive-segment speeds (robust to a short/behind first waypoint
        # and to per-frame model noise) rather than just |wp1-wp0|, which
        # under-estimates cruise speed and causes chronic light braking.
        if wp.shape[0] >= 2:
            segs = np.linalg.norm(np.diff(wp, axis=0), axis=1)   # (n-1,)
            vref_scalar = float(np.median(segs)) / self.wp_dt
        else:
            vref_scalar = np.linalg.norm(wp[0]) / self.wp_dt

        px = np.interp(tk, tw, wp[:, 0])
        py = np.interp(tk, tw, wp[:, 1])
        beyond = tk > tw[-1]
        if beyond.any():
            if wp.shape[0] >= 2:
                d = wp[-1] - wp[-2]
                n = np.linalg.norm(d)
                d = d / n if n > 1e-6 else np.array([1.0, 0.0])
            else:
                d = np.array([1.0, 0.0])
            ext = (tk[beyond] - tw[-1]) * max(vref_scalar, 1e-3)
            px[beyond] = wp[-1, 0] + d[0] * ext
            py[beyond] = wp[-1, 1] + d[1] * ext
        ref = np.stack([px, py], axis=1)
        vref = np.full(self.N, vref_scalar)
        return ref, vref

    # ------------------------------------------------------------------ #
    def _potential(self, P, obstacles):
        """Spatio-temporal potential + geometric collision check.
        Cost: smooth elliptical potential around each obstacle's FORECAST
        positions (prefers wider clearance).
        Feasibility: a candidate is infeasible only if it actually PENETRATES
        an obstacle's box inflated by the ego half-width + a hard safety
        margin, at the matching future time.
        Returns (U_cn (C,N) per-step potential, collide (C,) bool)."""
        C = P.shape[0]
        U_cn = np.zeros((C, self.N))
        collide = np.zeros(C, dtype=bool)
        tk = (np.arange(self.N) + 1) * self.dt
        for ob in obstacles:
            c = np.asarray(ob['center'], float)
            vv = np.asarray(ob.get('vel', (0.0, 0.0)), float)
            ex, ey = ob.get('extent', (2.4, 1.1))
            yaw = float(ob.get('yaw', 0.0))
            small = bool(ob.get('small', max(ex, ey) < self.vru_size))

            moving = float(np.hypot(vv[0], vv[1])) > 0.5
            A = self.pf_A * (self.vru_gain if small else 1.0)
            if not moving:
                A *= self.static_pf_frac
                # static obstacle sitting in another lane (large lateral
                # offset, not ahead in our corridor) -> no soft field at all;
                # the hard box still guarantees no contact.
                if abs(c[1]) > self.side_clear_lat and not small:
                    A = 0.0
            m_lat = self.pf_margin_lat + (self.vru_margin if small else 0.0)
            m_lon = self.pf_margin_long + (self.vru_margin if small else 0.0)
            # inflate the field along the obstacle's motion direction:
            # fast-moving obstacles cast a longer 'shadow' of risk.
            spd_ob = float(np.hypot(vv[0], vv[1]))
            infl = spd_ob * self.vel_inflate
            sx = ex + m_lon + self.ego_halflength
            sy = ey + m_lat + self.ego_halfwidth
            if spd_ob > 0.5:
                # motion mostly along obstacle-local x or y? project onto box axes
                cos0, sin0 = np.cos(yaw), np.sin(yaw)
                v_lx = abs(cos0 * vv[0] + sin0 * vv[1])
                v_ly = abs(-sin0 * vv[0] + cos0 * vv[1])
                den = v_lx + v_ly + 1e-6
                sx += infl * (v_lx / den)
                sy += infl * (v_ly / den)

            # hard box: obstacle extent + ego half-width + hard margin
            hard_m = self.hard_margin * (2.0 if small else 1.0)
            # Hard box = actual bumper-contact envelope. Longitudinally use the
            # ego FRONT reach (~1m), not the full half-length, so the ego can
            # legitimately close to a normal queue gap (2-3m center-to-center
            # beyond the boxes) instead of treating any car within 5m as a
            # collision. Lateral uses half-width for side passes.
            hx = ex + self.ego_front + hard_m
            hy = ey + self.ego_halfwidth + hard_m

            cos_, sin_ = np.cos(yaw), np.sin(yaw)
            oc = c[None, :] + vv[None, :] * tk[:, None]          # (N,2)
            dx = P[:, :, 0] - oc[None, :, 0]
            dy = P[:, :, 1] - oc[None, :, 1]
            lx = cos_ * dx + sin_ * dy
            ly = -sin_ * dx + cos_ * dy
            U = A * np.exp(-0.5 * ((lx / sx) ** 2 + (ly / sy) ** 2))  # (C,N)
            U_cn += U
            inside = (np.abs(lx) < hx) & (np.abs(ly) < hy)            # (C,N)
            # A candidate is a real collision only if it enters the hard box
            # while still carrying speed (would actually hit). Coming to rest
            # at the box edge (creeping up to a queue) is permitted.
            hit = inside & (self._Vref_for_collide > self.creep_speed)
            collide |= hit.any(axis=1)
        return U_cn, collide

    # ------------------------------------------------------------------ #
    def solve(self, v0, waypoints, obstacles):
        """Choose the best first control.
        Returns dict: delta (rad, +left), accel (m/s^2), feasible (bool),
        cost, debug fields."""
        P, V = self._rollout(v0)
        ref, vref = self._reference(waypoints)

        # dynamic feasibility: cap lateral acceleration (hard steer only at low
        # speed). a_lat = v^2 * tan(delta) / L per step.
        a_lat = (V ** 2) * (np.abs(np.tan(self._delta_kn)) / self.L)        # (C,N)
        dyn_infeasible = (a_lat > self.a_lat_max).any(axis=1)

        # Decompose deviation from the reference into lateral (across the
        # path) and longitudinal (along the path). Only LATERAL deviation is
        # a tracking error; longitudinal lag is handled gently by the
        # progress/speed terms, so braking is never punished as "off-path".
        tang = np.diff(np.vstack([[0.0, 0.0], ref]), axis=0)          # (N,2)
        tnorm = np.linalg.norm(tang, axis=1, keepdims=True)
        tang = np.where(tnorm > 1e-6, tang / np.maximum(tnorm, 1e-6),
                        np.array([1.0, 0.0]))
        dev = P - ref[None]                                           # (C,N,2)
        lat_dev = np.abs(dev[:, :, 0] * (-tang[None, :, 1])
                         + dev[:, :, 1] * tang[None, :, 0])           # (C,N)
        track = (lat_dev ** 2).sum(axis=1)
        lane = (np.clip(lat_dev - self.lane_bound, 0.0, None) ** 2).sum(axis=1)

        if obstacles:
            self._Vref_for_collide = V   # used by _potential's speed-gated collision
            U_cn, collide = self._potential(P, obstacles)
            # risk scales with ego speed: being NEAR an obstacle while slow or
            # stopped is safe (hard box still forbids contact); approaching it
            # fast is expensive. This is what lets the controller creep around
            # a stopped car instead of freezing at a distance.
            vfac = np.clip(V / self.pf_vscale, self.pf_vfloor, None)   # (C,N)
            Utot = (U_cn * vfac).sum(axis=1)
        else:
            Utot = np.zeros(P.shape[0]); collide = np.zeros(P.shape[0], dtype=bool)

        # Terminal blocked-state cost: a candidate that ENDS nearly stopped
        # with a static obstacle occupying the corridor just ahead is a
        # trapped state; price that into today's choice so commitment to a
        # feasible pass happens early, before the geometric window closes.
        term = np.zeros(P.shape[0])
        if obstacles:
            tang_end = tang[-1]
            Pf = P[:, -1, :]                                 # (C,2)
            Vf = V[:, -1]
            corridor_half = self.ego_halfwidth + 0.3
            blocked = np.zeros(P.shape[0], dtype=bool)
            for ob in obstacles:
                ovv = np.asarray(ob.get('vel', (0.0, 0.0)), float)
                if float(np.hypot(ovv[0], ovv[1])) > 0.5:
                    continue                                  # moving agents clear on their own
                oc = np.asarray(ob['center'], float)
                ex, ey = ob.get('extent', (2.4, 1.1))
                d = oc[None, :] - Pf                          # (C,2)
                ahead = d[:, 0] * tang_end[0] + d[:, 1] * tang_end[1]
                latoff = np.abs(-d[:, 0] * tang_end[1] + d[:, 1] * tang_end[0])
                blocked |= (ahead > 0.0) & (ahead < self.term_lookahead + ex) \
                           & (latoff < ey + corridor_half)
            term = np.where(blocked & (Vf < 1.5), self.w_term_blocked, 0.0)

        prog = -(np.minimum(V, vref[None]).sum(axis=1)) * self.dt   # negative = reward
        over = np.clip(V - vref[None] - self.speed_deadband, 0.0, None)
        under = np.clip(vref[None] - V, 0.0, None)
        spd = (over ** 2 + self.underspeed_frac * under ** 2).sum(axis=1)
        smooth = (self._delta - self.prev_first_delta) ** 2

        J = (self.w_track * track + self.w_lane * lane + self.w_pf * Utot
             + self.w_prog * prog + self.w_speed * spd + self.w_smooth * smooth
             + term)

        J = np.where(collide | dyn_infeasible, np.inf, J)

        if not np.isfinite(J).any():
            # every candidate would pass too close to something: hard brake
            self.prev_first_delta = 0.0
            return {'delta': 0.0, 'accel': self.a_min, 'feasible': False,
                    'cost': float('inf'), 'n_feasible': 0,
                    'straight_blocked': True}

        i = int(np.argmin(J))
        self.prev_first_delta = float(self._delta[i])

        # Diagnostic for steering-assist mode: is the model's own path (going
        # roughly straight while actually MOVING) geometrically blocked?
        # 'Straight blocked' asks: can the ego CONTINUE straight (still
        # moving at the horizon end)? The always-available stop-straight
        # candidate must not count as 'path clear'.
        straight = np.abs(self._delta) < 1e-9
        sustain = V[:, -1] > self.creep_speed
        sm = straight & sustain
        straight_blocked = bool(collide[sm].all()) if sm.any() else False

        return {'delta': float(self._delta[i]), 'accel': float(self._acc[i]),
                'feasible': True, 'cost': float(J[i]),
                'n_feasible': int(np.isfinite(J).sum()),
                'straight_blocked': straight_blocked}

    # ------------------------------------------------------------------ #
    def to_carla(self, sol, v0):
        """Map (delta, accel) to CARLA (steer, throttle, brake)."""
        steer = float(np.clip(self.steer_sign * sol['delta'] / self.delta_max, -1.0, 1.0))
        if v0 < 0.5 and sol['accel'] < -0.3:
            steer = 0.0   # no wheel-cranking while braking at standstill
        a = sol['accel']
        if a > 0.05:
            throttle = float(np.clip(a / max(self.a_max, 1e-6), 0.0, 1.0) * self.max_throttle)
            brake = 0.0
        elif a < -0.3:
            throttle = 0.0
            brake = float(np.clip(-a / (-self.a_min), 0.0, 1.0))
        else:
            throttle, brake = 0.0, 0.0
        return steer, throttle, brake


    # ------------------------------------------------------------------ #
    def steer_assist(self, v0, waypoints, obstacles):
        """Steering-only mode (option B).
        Longitudinal control (throttle/brake) stays entirely with the base
        controller (control_pid + the agent's safety box). The MPC proposes a
        steering override ONLY when the model's own path is geometrically
        blocked within the horizon AND a feasible steering maneuver exists.
        Everywhere else the base steering passes through untouched, so the
        default behavior is exactly the baseline.
        Returns dict: override (bool), delta (rad), reason, n_feasible."""
        if not obstacles:
            self._assist_hold = False
            return {'override': False, 'delta': 0.0,
                    'reason': 'no_obstacles', 'n_feasible': -1}

        # RULE 1 - VRU: never initiate a lateral avoidance around a cyclist or
        # pedestrian in our corridor. Braking (the base stack's job) is the
        # only acceptable response. This is the route-10/17 lesson.
        vru_ahead = any(o.get('small', False)
                        and 0.0 < o['center'][0] < 14.0
                        and abs(o['center'][1]) < 2.2 for o in obstacles)
        if vru_ahead and not getattr(self, '_assist_hold', False):
            return {'override': False, 'delta': 0.0,
                    'reason': 'vru_ahead_no_swerve', 'n_feasible': -1}

        # RULE 2 - HOLD hysteresis: once an avoidance starts, keep steering
        # with the MPC until the triggering region is actually CLEARED (no
        # non-VRU obstacle beside/ahead of us), not merely until the straight
        # path momentarily looks open. Releasing early hands control back to
        # the path-following PID, which would cut straight back into the
        # obstacle mid-pass.
        zone = any((not o.get('small', False))
                   and -2.0 < o['center'][0] < 12.0
                   and abs(o['center'][1]) < 4.5 for o in obstacles)
        hold = getattr(self, '_assist_hold', False)

        sol = self.solve(v0, waypoints, obstacles)

        if hold:
            # count how long we've been holding (for a hard timeout release)
            self._hold_ticks = getattr(self, '_hold_ticks', 0) + 1

            # RELEASE 1: the whole lateral zone is clear -> done passing.
            if not zone:
                self._assist_hold = False
                self._hold_ticks = 0
                return {'override': False, 'delta': 0.0,
                        'reason': 'cleared_release',
                        'n_feasible': sol['n_feasible']}
            # RELEASE 2 (NEW): the STRAIGHT path ahead is clear again. In dense
            # traffic the full lateral zone almost never empties, so the old
            # 'not zone' release would latch for the whole route and swerve.
            # Releasing when the straight path is open matches "I've passed the
            # thing I was avoiding" without waiting for an empty road.
            if not sol.get('straight_blocked', False):
                self._assist_hold = False
                self._hold_ticks = 0
                return {'override': False, 'delta': 0.0,
                        'reason': 'straight_clear_release',
                        'n_feasible': sol['n_feasible']}
            # RELEASE 3 (NEW): hard hold timeout. Never seize steering forever.
            if self._hold_ticks > getattr(self, 'assist_hold_max_ticks', 40):
                self._assist_hold = False
                self._hold_ticks = 0
                return {'override': False, 'delta': 0.0,
                        'reason': 'hold_timeout_release',
                        'n_feasible': sol['n_feasible']}
            if not sol['feasible']:
                # Emergency inside a pass: DO NOT hand steering back to the
                # path-following PID (it would cut straight into the obstacle
                # we are beside). Hold the current heading; the base stack's
                # braking stops us parallel to the obstacle instead.
                return {'override': True, 'delta': 0.0,
                        'reason': 'hold_straight_emergency', 'n_feasible': 0}
            # FIX B (NEW): if the solver agrees with ~straight, don't override
            # with a hard steer -- pass the model's steering through.
            if abs(sol['delta']) < 0.05:
                return {'override': False, 'delta': 0.0,
                        'reason': 'hold_agrees_straight',
                        'n_feasible': sol['n_feasible']}
            # FIX C (NEW): rate-limit the steering change so it cannot jump
            # full-lock left <-> right frame to frame (the +0.7/-0.5/+0.7 swerve).
            raw = float(sol['delta'])
            prev = getattr(self, '_hold_prev_delta', 0.0)
            max_step = getattr(self, 'assist_delta_rate', 0.10)  # rad per tick
            lim = max(prev - max_step, min(prev + max_step, raw))
            self._hold_prev_delta = lim
            return {'override': True, 'delta': lim,
                    'reason': 'hold', 'n_feasible': sol['n_feasible']}

        if not sol.get('straight_blocked', False):
            return {'override': False, 'delta': 0.0,
                    'reason': 'path_clear', 'n_feasible': sol['n_feasible']}
        if not sol['feasible']:
            return {'override': False, 'delta': 0.0,
                    'reason': 'no_feasible_maneuver', 'n_feasible': 0}
        if abs(sol['delta']) < 0.05:
            return {'override': False, 'delta': 0.0,
                    'reason': 'mpc_agrees_straight',
                    'n_feasible': sol['n_feasible']}
        self._assist_hold = True
        self._hold_ticks = 0
        self._hold_prev_delta = float(sol['delta'])
        return {'override': True, 'delta': sol['delta'],
                'reason': 'avoiding', 'n_feasible': sol['n_feasible']}

    def delta_to_steer(self, delta):
        """Map a steering angle (rad, module frame) to CARLA steer."""
        return float(np.clip(self.steer_sign * delta / self.delta_max,
                             -1.0, 1.0))


# ---------------------------------------------------------------------- #
def obstacles_from_bbs(curr_bbs, prev_bbs, ego_speed=0.0, frame_dt=0.1,
                       vru_size=1.0, match_dist=5.0):
    """Build the obstacle list from two consecutive detection frames.
    Each bb follows the agent's convention: bb[4] = center (x fwd, y left);
    extents from corner geometry; yaw from get_bb_yaw-equivalent outside.
    Velocity = displacement of the nearest match / frame_dt."""
    obstacles = []
    for bb in curr_bbs:
        cx, cy = float(bb[4, 0]), float(bb[4, 1])
        ex = 0.5 * np.sqrt((bb[3, 0] - bb[0, 0]) ** 2 + (bb[3, 1] - bb[0, 1]) ** 2)
        ey = 0.5 * np.sqrt((bb[0, 0] - bb[1, 0]) ** 2 + (bb[0, 1] - bb[1, 1]) ** 2)
        vx = vy = 0.0
        if prev_bbs is not None and len(prev_bbs) > 0:
            best, bd = None, match_dist ** 2
            for pb in prev_bbs:
                d = (float(pb[4, 0]) - cx) ** 2 + (float(pb[4, 1]) - cy) ** 2
                if d < bd:
                    bd, best = d, pb
            if best is not None:
                # bb differencing across ego frames measures RELATIVE velocity
                # (a parked car appears to move backward at ego_speed).
                # Compensate translation to recover the absolute velocity in
                # the current ego frame; rotation between frames is neglected.
                # relative velocity from bb differencing, then add ego motion
                rvx = (cx - float(best[4, 0])) / frame_dt
                rvy = (cy - float(best[4, 1])) / frame_dt
                # If the RELATIVE motion is tiny, the object is moving WITH the
                # world at ~0 absolute -- but bb differencing of a static car
                # across ego frames yields ~ -ego_speed relative, so the true
                # test for "static" is |rel + ego| small on each axis.
                vx = rvx + float(ego_speed)
                vy = rvy
                spd = np.hypot(vx, vy)
                # Defenses against ego-rotation / ID-switch corruption:
                # drop implausibly fast, and treat sub-noise as static.
                if spd > 8.0 or spd < 1.5:
                    vx = vy = 0.0
        # BUGFIX: orient the box - yaw from the long-edge direction so a
        # crossing car's hard box is not axis-aligned to the ego frame.
        yaw = float(np.arctan2(bb[3, 1] - bb[0, 1], bb[3, 0] - bb[0, 0]))
        # Cull obstacles that cannot matter: well behind the ego, or far to
        # the side. Keeps the cost focused on the corridor ahead and avoids
        # phantom braking from the clutter of a dense scene.
        if cx < -3.0 or abs(cy) > 8.0 or cx > 32.0:
            continue
        obstacles.append({'center': (cx, cy), 'vel': (vx, vy),
                          'extent': (float(ex), float(ey)), 'yaw': yaw,
                          'small': max(ex, ey) < vru_size})
    return obstacles
