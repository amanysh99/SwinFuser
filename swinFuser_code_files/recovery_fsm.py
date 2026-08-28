"""
RecoveryFSM v2 -- safe, curve-aware deadlock recovery for a SwinFuser/
TransFuser CARLA agent.

Why v2 (evidence from route 0, the failure the v1 FSM caused)
------------------------------------------------------------
v1 helped route 8 (removed collisions) but HURT route 0 (36 -> 22): it drove
into a guardrail on a curve and then burned the whole route in a retry loop.
Trace analysis of route 0 showed two distinct v1 failure modes:

  FM1 - FUTILE RETRY LOOP: the FSM re-triggered creep ~41 times at essentially
        the SAME location (e.g. 13 consecutive episodes at nearest_x=6.4),
        every one making zero progress, each timing out to COOLDOWN then
        re-arming. This wasted the entire route on an UNSOLVABLE deadlock.
        v1 gave up per-episode but never globally.

  FM2 - UNSAFE CREEP MOTION: when the creep DID move the car (vmax 2-3.5 m/s),
        it followed the model's waypoint heading at a forced speed with no
        curvature / lane awareness. On a curve that walked the car off-road
        into the guardrail. (control_pid steers toward waypoints even when
        is_stuck; the creep is NOT straight -- it follows the predicted path.)

v2 fixes both:

  Fix 1 (FM1) - GLOBAL give-up + PERMANENT_HOLD. Track re-trigger attempts at
        the same location; after `max_attempts_per_spot` failed attempts, enter
        PERMANENT_HOLD: stop creeping entirely, just hold.

  Fix 2 (FM2) - SAFETY-GATED creep. The FSM only REQUESTS a creep; the agent
        supplies `creep_safe` (from waypoint curvature + lateral deviation +
        obstacle centering). If unsafe, the FSM HOLDS instead of creeping, and
        bails to HOLD if the geometry turns unsafe mid-creep.

  Fix 3 - LESS EAGER trigger (~8 s vs v1's ~4 s).

States: NORMAL, CREEP_STRAIGHT, CREEP_STEER, HOLD_WAIT, PERMANENT_HOLD, COOLDOWN.
The FSM holds no CARLA handles and is deterministic.
"""
import math

NORMAL = "NORMAL"
CREEP_STRAIGHT = "CREEP_STRAIGHT"
CREEP_STEER = "CREEP_STEER"
HOLD_WAIT = "HOLD_WAIT"
PERMANENT_HOLD = "PERMANENT_HOLD"
COOLDOWN = "COOLDOWN"


def creep_safety_gate(pred_wp, nearest_x, nearest_y,
                      max_curve_rad=0.25, max_lat_dev=1.5,
                      obstacle_center_y=1.2, min_obstacle_x=1.5,
                      max_obstacle_x=10.0, stop_distance=5.0,
                      side_clear_y=2.0, in_path_stop=10.0):
    """Decide whether a forward creep is SAFE in the current geometry.

    Returns (safe: bool, reason: str). The agent calls this each tick and
    passes `safe` into RecoveryFSM.update(creep_safe=...).

    Safe ONLY when ALL hold:
      * predicted path ahead is nearly straight (curvature <= max_curve_rad).
        On a curve, forcing default_speed along curving waypoints walks the car
        off-lane -> the route-0 guardrail hit.
      * predicted path lateral deviation |y| <= max_lat_dev for every waypoint.
      * the blocker is roughly DEAD AHEAD (|nearest_y| < obstacle_center_y) and
        within [min_obstacle_x, max_obstacle_x].

    pred_wp : (N,2) array, ego frame, x forward, y left. If None -> unsafe.
    """
    import numpy as np
    if pred_wp is None or len(pred_wp) < 2:
        return False, 'no_waypoints'
    wp = np.asarray(pred_wp, dtype=float)

    near = wp[1] - wp[0]
    far = wp[-1] - wp[-2]
    h_near = math.atan2(near[1], near[0])
    h_far = math.atan2(far[1], far[0])
    dh = abs(math.atan2(math.sin(h_far - h_near), math.cos(h_far - h_near)))
    if dh > max_curve_rad:
        return False, 'curve_%.2f' % dh

    if np.abs(wp[:, 1]).max() > max_lat_dev:
        return False, 'lateral_%.1f' % float(np.abs(wp[:, 1]).max())

    if nearest_x is None:
        return False, 'no_obstacle'
    # HARD STOP: never creep toward a car that is closer than the safe
    # following distance AND in our path. This is the fix for "the car creeps
    # until it hits the vehicle in front" -- as long as a vehicle is within
    # stop_distance ahead AND roughly in our lane, forward creep is forbidden
    # (the FSM will HOLD and wait instead).
    # in_path_stop is LONGER than stop_distance: a big vehicle (truck) sitting
    # dead-ahead at x=7.2 was let through because stop_distance=5.0 < 7.2, so
    # CREEP_STEER drove into it. Any obstacle IN OUR CORRIDOR within
    # in_path_stop blocks the creep, regardless of the shorter stop_distance.
    if abs(nearest_y) < obstacle_center_y and nearest_x < in_path_stop:
        return False, 'in_path_%.1f' % nearest_x
    if abs(nearest_y) < obstacle_center_y and nearest_x < stop_distance:
        return False, 'too_close_%.1f' % nearest_x
    # SIDE obstacle: if the blocker is clearly OFF TO THE SIDE (outside the ego
    # corridor), a forward creep does NOT drive into it -> creeping is SAFE.
    # The old code returned unsafe here ('obstacle_offset'), which made the car
    # HOLD forever for a motorcycle/car sitting beside it (|y|=2.6) even though
    # the path ahead was clear -> PERMANENT_HOLD / timeout. A side obstacle is
    # a reason to GO, not to freeze. side_clear_y marks "outside our corridor".
    if abs(nearest_y) >= side_clear_y:
        return True, 'safe_side_clear_%.1f' % nearest_y
    if not (min_obstacle_x < nearest_x < max_obstacle_x):
        return False, 'obstacle_range_%.1f' % nearest_x
    if abs(nearest_y) > obstacle_center_y:
        # in the ambiguous band (obstacle_center_y < |y| < side_clear_y): the
        # blocker is near the edge of our corridor -- creep only if the path
        # ahead is straight (already checked above) and treat as safe-ish.
        return True, 'safe_edge_%.1f' % nearest_y

    return True, 'safe'


class RecoveryFSM:
    def __init__(self, config=None):
        g = (lambda k, d: getattr(config, k, d)) if config is not None else (lambda k, d: d)

        self.action_repeat = int(g('action_repeat', 2))

        # --- Trigger ---
        self.stall_speed = float(g('rec_stall_speed', 0.1))
        self.stall_ticks_to_creep = int(g('rec_stall_ticks', 80))   # Fix 3: ~8 s

        # --- Phase budgets (agent ticks) ---
        self.creep_straight_ticks = int(g('rec_creep_straight_ticks', 10))
        self.creep_steer_ticks = int(g('rec_creep_steer_ticks', 40))
        self.hold_wait_ticks = int(g('rec_hold_wait_ticks', 60))
        self.cooldown_ticks = int(g('rec_cooldown_ticks', 30))

        # --- Creep speeds ---
        self.creep_speed = float(g('rec_creep_speed', 1.5))   # gentler than v1
        self.escaped_speed = float(g('rec_escaped_speed', 1.5))
        self.escape_confirm_ticks = int(g('rec_escape_confirm_ticks', 5))
        self.moving_confirm_ticks = int(g('rec_moving_confirm_ticks', 8))

        # --- Give-up ---
        self.max_recovery_ticks = int(g('rec_max_ticks', 100))
        self.max_attempts_per_spot = int(g('rec_max_attempts_per_spot', 3))  # Fix 1
        self.perm_hold_clear_ticks = int(g('rec_perm_hold_clear_ticks', 15))  # exit PERMANENT_HOLD when path clear this long
        # nudge when stalled with NOTHING ahead (must stay above red-light wait)
        self.free_stall_ticks = int(g('rec_free_stall_ticks', 500))
        self.same_spot_radius = float(g('rec_same_spot_radius', 8.0))
        self.progress_reset_dist = float(g('rec_progress_reset_dist', 10.0))

        self.reset()

    def reset(self):
        self.state = NORMAL
        self.stall_counter = 0
        self.phase_counter = 0
        self.total_recovery_ticks = 0
        self.cooldown_counter = 0
        self.hold_counter = 0
        self.escape_confirm = 0
        self.moving_confirm = 0
        self.attempt_spot = None
        self.attempt_count = 0
        self.perm_hold_counter = 0
        self.perm_clear_confirm = 0

    # ------------------------------------------------------------------ #
    def update(self, ego_speed, obstacle_ahead, moving_obstacle_ahead,
               made_progress, creep_safe=True, ego_pos=None):
        stalled = ego_speed < self.stall_speed

        # ---------------- PERMANENT_HOLD ----------------
        if self.state == PERMANENT_HOLD:
            self.perm_hold_counter += 1
            if made_progress and ego_speed > self.escaped_speed:
                self._rearm()
                return self._out(NORMAL, False, 0.0, False, 'perm_hold_resolved')
            # ESCAPE when the path CLEARS. PERMANENT_HOLD was designed for a real
            # deadlock (something blocking us). But the old code's ONLY exit was
            # "made_progress + speed" -- impossible while stopped -> once the car
            # entered PERMANENT_HOLD it stayed FROZEN FOREVER even after the
            # blocker drove away (log: PERMANENT_HOLD, nobs=0, nearest=None,
            # v=0). If no obstacle is ahead for a short confirm window, release
            # to NORMAL so the car resumes following its waypoints.
            if not obstacle_ahead:
                self.perm_clear_confirm = getattr(self, 'perm_clear_confirm', 0) + 1
                if self.perm_clear_confirm >= self.perm_hold_clear_ticks:
                    self._rearm()
                    self.perm_clear_confirm = 0
                    return self._out(NORMAL, False, 0.0, False, 'perm_hold_path_clear')
            else:
                self.perm_clear_confirm = 0
            return self._out(PERMANENT_HOLD, False, 0.0, False, 'perm_hold')

        # ---------------- COOLDOWN ----------------
        if self.state == COOLDOWN:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.cooldown_ticks:
                self.state = NORMAL
                self.stall_counter = 0
            return self._out(NORMAL, False, 0.0, False, 'cooldown')

        # ---------------- NORMAL ----------------
        if self.state == NORMAL:
            if stalled:
                self.stall_counter += 1
            else:
                self.stall_counter = 0
                self._maybe_reset_attempts(ego_pos)

            deadlocked = (self.stall_counter >= self.stall_ticks_to_creep
                          and obstacle_ahead
                          and not moving_obstacle_ahead)
            # FREE STALL: stopped for a long time with NOTHING ahead. The old
            # code required obstacle_ahead, so this case was ignored ENTIRELY --
            # the car sat still until BLOCKED/timeout while the FSM reported
            # 'normal, is_stuck=False' (route 8: stopped 63% of the run, 46% of
            # it with no obstacle, recovery triggered 0 times).
            # This is also the SAFEST case to nudge: nothing is in front to hit.
            # The threshold is LONGER than the blocked-by-vehicle one so we
            # never nudge through a red light (where there is also no vehicle
            # ahead) -- same rationale as stall_ticks_to_creep.
            # DISABLED by default (free_stall_ticks <= 0). MEASURED RESULT: this
            # nudge cost -25.8 DS over 5 routes. Two failure modes, both from the
            # same wrong assumption ("nothing ahead" == "safe to move"):
            #   * a RED LIGHT also has no vehicle ahead -> the nudge ran the
            #     light (route 3), and a queue at a light easily exceeds 50 s.
            #   * obstacle_ahead uses a NARROW box (rec_trigger_x ~8 m, |y|<2.5),
            #     so a truck at 9 m or slightly off-centre reads as "nothing",
            #     and the nudge drove into it -- route 4 hit the SAME vehicle 3x.
            # Keep the code (documented negative result) but leave it off unless
            # obstacle_ahead is made as wide as the waypoint-safety check AND a
            # real traffic-light signal is available to gate it.
            free_stall = (self.free_stall_ticks > 0
                          and self.stall_counter >= self.free_stall_ticks
                          and not obstacle_ahead)
            if not (deadlocked or free_stall):
                return self._out(NORMAL, False, 0.0, False, 'normal')

            # GLOBAL give-up (Fix 1)
            if self._same_spot(ego_pos):
                self.attempt_count += 1
            else:
                self.attempt_spot = ego_pos
                self.attempt_count = 1
            if self.attempt_count > self.max_attempts_per_spot:
                self.state = PERMANENT_HOLD
                self.perm_hold_counter = 0
                return self._out(PERMANENT_HOLD, False, 0.0, False,
                                 'giveup_spot_x%d' % self.attempt_count)

            self.total_recovery_ticks = 0
            if not creep_safe:                       # Fix 2
                self.state = HOLD_WAIT
                self.hold_counter = 0
                return self._out(HOLD_WAIT, False, 0.0, False, 'enter_hold_unsafe')
            self.state = CREEP_STRAIGHT
            self.phase_counter = 0
            return self._out(CREEP_STRAIGHT, True, self.creep_speed, False,
                             'enter_creep_free' if (free_stall and not deadlocked)
                             else 'enter_creep_straight')

        # ---------------- recovery active ----------------
        self.total_recovery_ticks += 1

        if made_progress and ego_speed > self.escaped_speed and not obstacle_ahead:
            self.escape_confirm += 1
        else:
            self.escape_confirm = 0
        if self.escape_confirm >= self.escape_confirm_ticks:
            self._rearm()
            self._enter_cooldown()
            return self._out(COOLDOWN, False, 0.0, False, 'escaped')

        if self.total_recovery_ticks >= self.max_recovery_ticks:
            self._enter_cooldown()
            return self._out(COOLDOWN, False, 0.0, False, 'episode_budget')

        # Fix 2: bail if geometry turned unsafe mid-creep
        if self.state in (CREEP_STRAIGHT, CREEP_STEER) and not creep_safe:
            self.state = HOLD_WAIT
            self.hold_counter = 0
            return self._out(HOLD_WAIT, False, 0.0, False, 'creep_became_unsafe')

        if moving_obstacle_ahead:
            self.moving_confirm += 1
            if self.moving_confirm >= self.moving_confirm_ticks:
                self._enter_cooldown()
                return self._out(COOLDOWN, False, 0.0, False, 'moving_blocker_abort')
            return self._out(self.state, False, 0.0,
                             self.state == CREEP_STEER, 'moving_flicker_wait')
        else:
            self.moving_confirm = 0

        # ---------------- HOLD_WAIT ----------------
        if self.state == HOLD_WAIT:
            self.hold_counter += 1
            if self.hold_counter >= self.hold_wait_ticks:
                self._enter_cooldown()
                return self._out(COOLDOWN, False, 0.0, False, 'hold_wait_done')
            return self._out(HOLD_WAIT, False, 0.0, False, 'hold_wait')

        # ---------------- CREEP_STRAIGHT ----------------
        if self.state == CREEP_STRAIGHT:
            self.phase_counter += 1
            if self.phase_counter >= self.creep_straight_ticks:
                self.state = CREEP_STEER
                self.phase_counter = 0
                return self._out(CREEP_STEER, True, self.creep_speed, True,
                                 'escalate_creep_steer')
            return self._out(CREEP_STRAIGHT, True, self.creep_speed, False,
                             'creep_straight')

        # ---------------- CREEP_STEER ----------------
        if self.state == CREEP_STEER:
            self.phase_counter += 1
            if self.phase_counter >= self.creep_steer_ticks:
                self._enter_cooldown()
                return self._out(COOLDOWN, False, 0.0, False, 'creep_steer_timeout')
            return self._out(CREEP_STEER, True, self.creep_speed, True,
                             'creep_steer')

        return self._out(NORMAL, False, 0.0, False, 'fallback')

    # ------------------------------------------------------------------ #
    def _same_spot(self, ego_pos):
        if ego_pos is None or self.attempt_spot is None:
            return self.attempt_spot is not None
        dx = ego_pos[0] - self.attempt_spot[0]
        dy = ego_pos[1] - self.attempt_spot[1]
        return (dx * dx + dy * dy) ** 0.5 < self.same_spot_radius

    def _maybe_reset_attempts(self, ego_pos):
        if ego_pos is None or self.attempt_spot is None:
            return
        dx = ego_pos[0] - self.attempt_spot[0]
        dy = ego_pos[1] - self.attempt_spot[1]
        if (dx * dx + dy * dy) ** 0.5 > self.progress_reset_dist:
            self.attempt_spot = None
            self.attempt_count = 0

    def _rearm(self):
        self.attempt_spot = None
        self.attempt_count = 0

    def _enter_cooldown(self):
        self.state = COOLDOWN
        self.cooldown_counter = 0
        self.stall_counter = 0
        self.phase_counter = 0
        self.hold_counter = 0
        self.escape_confirm = 0
        self.moving_confirm = 0

    def _out(self, mode, creep, target_speed, allow_mpc_steer, reason):
        return {'mode': mode, 'creep': creep, 'target_speed': target_speed,
                'allow_mpc_steer': allow_mpc_steer, 'reason': reason,
                'state': self.state}
