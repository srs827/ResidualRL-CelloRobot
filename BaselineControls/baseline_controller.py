"""

Force-controlled baseline controller for target string (in this case we use A string).
Parses a MIDI file and translates all notes to target string.
Controls dynamics, note length, speed, and bow direction via RTDE.
Goal: perfect single-string such that we can reach proper expression, bowings, etc. for the 
      entire MIDI sequence.

Features:
    - Parse MIDI -> note events (all mapped to target string)
    - Maintain stroke trajectory (interpolated frog <-> tip)
    - Execute via servoL (position) — force approximated via z-offset
    - Log full measured state continuously at 100Hz
    - Expose get_baseline_action() / execute_timestep() / get_full_state()

"""

import numpy as np
import time
import threading
import mido
import rtde_control
import rtde_receive
import atexit

# Robot connection 
ROBOT_IP = "192.168.1.100"

# TCP offset and payload: needs to match target string script exactly 
TCP_OFFSET  = [0.028210348281514253, -0.09610723587300697,
               -0.09969041498611403, 0.0, 0.0, 0.0]
PAYLOAD_KG  = 0.190000
PAYLOAD_COG = [0.128, -0.006, 0.019] # center of gravity 

# Target string waypoints (from a_full_bow.script here)
FROG_TARGET_STR = np.array([.336637615375,  .773335607743,  .101937252349,
                   -1.369835518384, -2.336199267621, 1.326965437172])
TIP_TARGET_STR  = np.array([.525205911288,  .350983193771,  .212779688012,
                   -1.369835518377, -2.336199267606, 1.326965437192])

BOW_LENGTH_TARGET_STR = float(np.linalg.norm(TIP_TARGET_STR[:3] - FROG_TARGET_STR[:3]))  # meters

# Force control calibration for target string 
FORCE_MIN   = 1.8    # N, lightest usable tone (just above flautando)
FORCE_MAX   = 5.5    # N, loudest before raucous
SPEED_MIN   = 0.04   # m/s
SPEED_MAX   = 0.15   # m/s

# Force mode constants (kept for reference, force mode currently disabled)
TASK_FRAME_A      = [0.30072696597818693, 0.793561615546182, 0.09970470799022213,
                     -1.54401381077418, -2.354969999658093, 1.3469994779985106]
SELECTION_VECTOR  = [0, 0, 1, 0, 0, 0]
FORCE_MODE_LIMITS = [0.1, 0.1, 0.05, 0.1, 0.1, 0.1]
FORCE_MODE_TYPE   = 2

# Timing
TIMESTEP_SEC  = 0.05   # 50ms per RL/baseline step
STATE_LOG_HZ  = 100    # F/T sensor logging rate during strokes


# ── Dynamics mapping ──────────────────────────────────────────────

def velocity_to_dynamic(velocity: int) -> str:
    """MIDI velocity (0-127) → dynamic marking string."""
    if velocity < 32:  return 'pp'
    if velocity < 48:  return 'mp'
    if velocity < 64:  return 'mf'
    if velocity < 96:  return 'f'
    return 'ff'


def velocity_to_force(velocity: int) -> float:
    """MIDI velocity -> bow force (N). Louder = more force."""
    v_norm = np.clip(velocity / 127.0, 0.0, 1.0)
    return float(FORCE_MIN + v_norm * (FORCE_MAX - FORCE_MIN))


def velocity_to_speed(velocity: int) -> float:
    """MIDI velocity -> bow speed (m/s). Louder = faster bow."""
    v_norm = np.clip(velocity / 127.0, 0.0, 1.0)
    return float(SPEED_MIN + v_norm * (SPEED_MAX - SPEED_MIN))


# ── MIDI parsing ──────────────────────────────────────────────────

def parse_midi_a_string(midi_path: str) -> list:
    """
    Parse MIDI file and return note events, all mapped to A string.

    All pitch information is ignored — every note becomes an open A.
    Duration and velocity (dynamics) are preserved.
    Bow direction alternates every note.

    Returns list of dicts:
        duration:  float (seconds)
        velocity:  int   (1-127)
        dynamic:   str   ('pp','mp','mf','f','ff')
        bow_dir:   str   ('down' or 'up')
        string:    str   ('A')
    """
    mid    = mido.MidiFile(midi_path)
    tempo  = 500000  # default 120 BPM (microseconds per beat)

    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    ticks_per_beat = mid.ticks_per_beat

    def ticks_to_sec(ticks):
        return ticks * tempo / (ticks_per_beat * 1_000_000)

    events       = []
    active_notes = {}
    current_tick = 0
    bow_dir      = 'down'

    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (current_tick, msg.velocity)

            elif msg.type in ('note_off', 'note_on') and msg.note in active_notes:
                start_tick, velocity = active_notes.pop(msg.note)
                duration = ticks_to_sec(current_tick - start_tick)

                if duration < 0.05:
                    continue

                events.append({
                    'duration': float(duration),
                    'velocity': int(velocity),
                    'dynamic':  velocity_to_dynamic(velocity),
                    'bow_dir':  bow_dir,
                    'string':   'A',
                })

                bow_dir = 'up' if bow_dir == 'down' else 'down'

    print(f"Parsed {len(events)} notes from {midi_path}")
    for i, e in enumerate(events[:8]):
        print(f"  [{i+1}] dur={e['duration']:.2f}s  vel={e['velocity']:3d}  "
              f"dyn={e['dynamic']:2s}  dir={e['bow_dir']}")
    if len(events) > 8:
        print(f"  ... ({len(events) - 8} more)")

    return events


# ── State logger ──────────────────────────────────────────────────

class StateLogger:
    """
    Logs RTDE state continuously at STATE_LOG_HZ during a stroke.
    Runs in a background thread so it doesn't block audio recording.

    Usage:
        logger.start()
        # ... robot executes stroke ...
        logger.stop()
        summary = logger.get_summary()
        logger.save('stroke_001_state.npy')
    """

    def __init__(self, rtde_r, hz: int = STATE_LOG_HZ,
                 frog=None, tip=None):
        self.rtde_r   = rtde_r
        self.interval = 1.0 / hz
        self.log      = []
        self._running = False
        self._thread  = None
        # Bow reference waypoints for bow_pos computation. Default to the
        # module-level constants; callers should override via set_bow_reference()
        # so bow_pos is computed on the exact axis the recording script uses.
        self._frog = np.asarray(frog, dtype=float) if frog is not None \
                     else FROG_TARGET_STR.copy()
        self._tip  = np.asarray(tip,  dtype=float) if tip  is not None \
                     else TIP_TARGET_STR.copy()

    def set_bow_reference(self, frog, tip):
        """
        Update the frog/tip waypoints used for bow_pos projection.
        Call before each condition when the reference frame changes
        (e.g. different angle in the tilt/yaw experiment, or to sync
        with the recording script's waypoints which differ from the
        controller's by 2mm in z).
        Safe to call while the logger is stopped between strokes.
        """
        self._frog = np.asarray(frog, dtype=float)
        self._tip  = np.asarray(tip,  dtype=float)

    def start(self):
        self.log      = []
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        while self._running:
            try:
                tcp    = self.rtde_r.getActualTCPPose()
                ft     = self.rtde_r.getActualTCPForce()
                joints = self.rtde_r.getActualQ()
                vel    = self.rtde_r.getActualTCPSpeed()

                bow_axis = self._tip[:3] - self._frog[:3]
                t_pos    = np.dot(np.array(tcp[:3]) - self._frog[:3], bow_axis)
                t_pos   /= (np.dot(bow_axis, bow_axis) + 1e-12)
                bow_pos  = float(np.clip(t_pos, 0.0, 1.0))

                bow_axis_norm = bow_axis / (np.linalg.norm(bow_axis) + 1e-8)
                bow_speed     = float(np.dot(np.array(vel[:3]), bow_axis_norm))

                self.log.append({
                    't':         time.time(),
                    # TCP pose
                    'tcp_x':  tcp[0], 'tcp_y':  tcp[1], 'tcp_z':  tcp[2],
                    'tcp_rx': tcp[3], 'tcp_ry': tcp[4], 'tcp_rz': tcp[5],
                    # F/T sensor
                    'ft_fx':  ft[0],  'ft_fy':  ft[1],  'ft_fz':  ft[2],
                    'ft_tx':  ft[3],  'ft_ty':  ft[4],  'ft_tz':  ft[5],
                    # Joint angles
                    'j0': joints[0], 'j1': joints[1], 'j2': joints[2],
                    'j3': joints[3], 'j4': joints[4], 'j5': joints[5],
                    # Derived
                    'bow_position': bow_pos,
                    'bow_speed':    bow_speed,
                })
            except Exception:
                pass

            time.sleep(self.interval)

    def get_summary(self, t_start=None, t_end=None) -> dict:
        """
        Summary statistics over the logged stroke window.
        Written to metadata.jsonl.

        Force/contact-magnitude fields have been removed: the loose bow mount
        and gravity-comp residual make them unreliable (constant 0.4–0.6 N
        regardless of depth). Use torque_contact_est_N instead.

        Joint keys (j3_mid, j4_mid, j5_mid) use 0-based indexing to match the
        log keys ('j3', 'j4', 'j5' = joints[3,4,5] = UR5e wrist joints).
        """
        log = self.log
        if t_start is not None and t_end is not None:
            log = [s for s in log if t_start <= s['t'] <= t_end]
        if not log:
            return {}

        spd = np.array([s['bow_speed']    for s in log])
        pos = np.array([s['bow_position'] for s in log])
        tx  = np.array([s['ft_tx']        for s in log])
        ty  = np.array([s['ft_ty']        for s in log])
        tz  = np.array([s['ft_tz']        for s in log])

        torque_mag = np.sqrt(tx**2 + ty**2 + tz**2)

        # Moment arm from F/T sensor origin to bow-hair contact point.
        # Derived from TCP_OFFSET = [0.028, -0.096, -0.100] m → |arm| ≈ 0.140 m.
        # Physically measure frog/flange-to-hair on your actual mount and update.
        MOMENT_ARM_M = 0.140

        mid = len(log) // 2

        return {
            # ── Bow speed ────────────────────────────────────────────
            # speed_mean includes accel/decel ramps; speed_max ≈ cruise speed.
            # Use speed_max as the honest "commanded speed was reached" check.
            'speed_mean':             float(np.mean(np.abs(spd))),
            'speed_std':              float(np.std(spd)),
            'speed_min':              float(np.min(np.abs(spd))),
            'speed_max':              float(np.max(np.abs(spd))),
            'speed_p10':              float(np.percentile(np.abs(spd), 10)),
            'speed_p90':              float(np.percentile(np.abs(spd), 90)),

            # ── Torque-based contact force estimate ──────────────────
            # |τ| after per-condition tare reflects contact-induced torque.
            # torque_contact_est_N = mean(|τ|) / moment_arm. More sensitive
            # than |F| because the loose mount's noise is primarily in the
            # force channels, not the torque channels.
            # Validate by plotting torque_contact_est_N vs depth_m — should
            # be monotonic if the moment arm assumption holds.
            'torque_mag_mean':        float(np.mean(torque_mag)),
            'torque_mag_std':         float(np.std(torque_mag)),
            'torque_mag_max':         float(np.max(torque_mag)),
            'torque_contact_est_N':   float(np.mean(torque_mag) / MOMENT_ARM_M),

            # ── Bow position ─────────────────────────────────────────
            # Fraction along self._frog -> self._tip axis (set via
            # set_bow_reference() before each condition). Sign of bow_pos_range
            # encodes direction: negative = up-bow, positive = down-bow.
            'bow_pos_start':          float(pos[0]),
            'bow_pos_end':            float(pos[-1]),
            'bow_pos_range':          float(pos[-1] - pos[0]),

            # ── Wrist joints at stroke midpoint ──────────────────────
            # 0-indexed to match log keys: j3/j4/j5 = joints[3/4/5] = UR5e wrist.
            'j3_mid':                 float(self.log[mid]['j3']),
            'j4_mid':                 float(self.log[mid]['j4']),
            'j5_mid':                 float(self.log[mid]['j5']),

            # ── TCP at stroke midpoint ───────────────────────────────
            # Basis-independent position; use for cross-script comparisons
            # instead of bow_pos when frog/tip reference differs.
            'tcp_x_mid':              float(self.log[mid]['tcp_x']),
            'tcp_y_mid':              float(self.log[mid]['tcp_y']),
            'tcp_z_mid':              float(self.log[mid]['tcp_z']),

            'n_samples':              len(self.log),
        }

    def get_current_state(self) -> dict:
        """Return most recent logged state (for real-time RL use)."""
        if not self.log:
            return {}
        return self.log[-1]

    def save(self, path: str):
        """Save full time series to .npy for later analysis."""
        np.save(path, self.log)


# ── Main baseline controller ──────────────────────────────────────

class TargetStringBaselineController:
    """
    Baseline controller for target string.

    Interface:
        reset()                      -> move to frog, ready to begin
        get_baseline_action()        -> dict with tcp_target, force_nominal, etc.
        execute_timestep(tcp, force) -> send servoL command
        get_full_state()             -> dict of measured RTDE values
        get_score_context()          -> dict of current MIDI position
        is_complete()                -> bool
        close()                      -> clean shutdown
    """

    SERVO_VEL        = 0.1     # m/s
    SERVO_ACCEL      = 0.1     # m/s^2
    SERVO_LOOKAHEAD  = 0.1     # seconds
    SERVO_GAIN       = 300
    Z_PER_NEWTON     = 0.0005  # m/N: do not change without re-testing
    BASELINE_FORCE_A = 3.0     # N nominal contact force

    def __init__(self, note_events: list):
        self.note_events = note_events

        print(f"Connecting to {ROBOT_IP}...")
        self.rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

        atexit.register(self.rtde_c.disconnect)
        atexit.register(self.rtde_r.disconnect)

        self.rtde_c.setTcp(TCP_OFFSET)
        self.rtde_c.setPayload(PAYLOAD_KG, PAYLOAD_COG)

        print("Connected.")
        print(f"  TCP pose:  {self.rtde_r.getActualTCPPose()}")
        print(f"  TCP force: {self.rtde_r.getActualTCPForce()}")
        print(f"  Joints:    {self.rtde_r.getActualQ()}")

        self.state_logger = StateLogger(self.rtde_r)

        self.current_note_idx   = 0
        self.current_step       = 0
        self._trajectory        = []
        self._n_steps_this_note = 0
        self._current_force     = 0.0
        self._current_note      = {}

    def reset(self):
        """Reset to start of sequence. Moves robot to frog position."""
        self.current_note_idx = 0
        self.current_step     = 0
        self._trajectory      = []
        self._current_note    = {}

        print("Moving to frog position...")
        self.rtde_c.moveL(list(FROG_TARGET_STR), 0.1, 0.5)
        time.sleep(0.5)

        # Check contact after moving to frog
        ft  = self.rtde_r.getActualTCPForce()
        mag = float(np.linalg.norm(ft[:3]))
        if mag < 0.3:
            print(f"  Low contact force: {mag:.3f}N : bow may not be on string")
        else:
            print(f"  Contact: {mag:.3f}N")

        print("Ready at frog.")

    def get_baseline_action(self) -> dict | None:
        """
        Returns nominal action for current timestep.
        Call every 50ms. Returns None when sequence is complete.
        """
        if self.current_note_idx >= len(self.note_events):
            return None

        if self.current_step == 0:
            self._setup_note(self.current_note_idx)

        if self.current_step >= self._n_steps_this_note:
            self.current_note_idx += 1
            self.current_step      = 0

            if self.current_note_idx >= len(self.note_events):
                return None

            self._setup_note(self.current_note_idx)

        note    = self.note_events[self.current_note_idx]
        tcp     = self._trajectory[self.current_step]
        bow_pos = self.current_step / max(self._n_steps_this_note - 1, 1)

        self._current_note = note
        self.current_step += 1

        return {
            'tcp_target':    tcp,
            'force_nominal': velocity_to_force(note['velocity']),
            'bow_position':  float(bow_pos),
            'bow_direction': note['bow_dir'],
            'string':        'A',
            'is_transition': False,
            'step':          self.current_step,
            'note_duration': note['duration'],
            'velocity':      note['velocity'],
            'dynamic':       note['dynamic'],
        }

    def execute_timestep(self, tcp_target: np.ndarray, force: float):
        """
        Send one timestep command via servoL.
        Force is approximated via z-offset from nominal position.
        forceMode is disabled : conflicts with servoL on UR5e.
        """
        z_offset = (force - self.BASELINE_FORCE_A) * self.Z_PER_NEWTON
        tcp = tcp_target.copy()
        tcp[2] += z_offset

        self.rtde_c.servoL(
            tcp.tolist(),
            self.SERVO_VEL,
            self.SERVO_ACCEL,
            TIMESTEP_SEC,
            self.SERVO_LOOKAHEAD,
            self.SERVO_GAIN,
        )

    def get_full_state(self) -> dict:
        """
        Returns current measured robot state from RTDE.
        Called by RL environment to build observation.
        """
        tcp    = np.array(self.rtde_r.getActualTCPPose())
        ft     = np.array(self.rtde_r.getActualTCPForce())
        joints = np.array(self.rtde_r.getActualQ())
        vel    = np.array(self.rtde_r.getActualTCPSpeed())

        bow_axis      = TIP_TARGET_STR[:3] - FROG_TARGET_STR[:3]
        t_pos         = np.dot(tcp[:3] - FROG_TARGET_STR[:3], bow_axis)
        t_pos        /= (np.dot(bow_axis, bow_axis) + 1e-12)
        bow_pos       = float(np.clip(t_pos, 0.0, 1.0))

        bow_axis_norm = bow_axis / (np.linalg.norm(bow_axis) + 1e-8)
        bow_speed     = float(np.dot(vel[:3], bow_axis_norm))

        note = self._current_note or {}

        contact_mag = float(np.linalg.norm(ft[:3]))

        return {
            # Full sensor readings
            'tcp':                     tcp,
            'ft':                      ft,
            'joints':                  joints,
            'tcp_velocity':            vel,

            # Bow kinematics
            'bow_position':            bow_pos,
            'bow_speed':               bow_speed,
            'bow_direction':           1.0 if note.get('bow_dir') == 'down' else -1.0,

            #  both fz and total magnitude
            # fz alone is unreliable, contact force spreads across fx/fy/fz
            # use contact_force_magnitude as primary force feature
            'measured_force':          float(ft[2]),       # fz component only
            'contact_force_magnitude': contact_mag,         # total 

            # deviation 
            'force_deviation':         contact_mag - self._current_force,

            # Contact quality indicators
            'lateral_force':           float(ft[0]),
            'torque_x':                float(ft[3]),

            'bow_accel':               0.0,

            # Wrist joints
            'j4':                      float(joints[3]),
            'j5':                      float(joints[4]),
            'j6':                      float(joints[5]),
        }

    def get_score_context(self) -> dict:
        """Current MIDI score position for RL observation."""
        if self.current_note_idx >= len(self.note_events):
            return {'sequence_done': True}

        note    = self.note_events[self.current_note_idx]
        bow_pos = self.current_step / max(self._n_steps_this_note, 1)

        next_note = (
            self.note_events[self.current_note_idx + 1]
            if self.current_note_idx + 1 < len(self.note_events)
            else None
        )

        return {
            'current_string':  'A',
            'dynamic':         note['dynamic'],
            'velocity':        note['velocity'],
            'note_duration':   note['duration'],
            'bow_direction':   note['bow_dir'],
            'bow_position':    float(bow_pos),
            'time_remaining':  float((1.0 - bow_pos) * note['duration']),
            'note_idx':        self.current_note_idx,
            'n_notes':         len(self.note_events),
            'next_dynamic':    next_note['dynamic']  if next_note else 'mf',
            'next_velocity':   next_note['velocity'] if next_note else 64,
            'next_bow_dir':    next_note['bow_dir']  if next_note else 'up',
            'sequence_done':   False,
        }

    def is_complete(self) -> bool:
        return self.current_note_idx >= len(self.note_events)

    def close(self):
        """Clean shutdown."""
        self.state_logger.stop()
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass
        try:
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
        except Exception:
            pass
        print("Controller closed.")


    def _setup_note(self, idx: int):
        """Compute trajectory for note at idx. Called once per note."""
        note    = self.note_events[idx]
        force   = velocity_to_force(note['velocity'])
        speed   = velocity_to_speed(note['velocity'])
        n_steps = max(2, int(note['duration'] / TIMESTEP_SEC))

        distance     = min(speed * note['duration'], BOW_LENGTH_TARGET_STR)
        bow_fraction = distance / BOW_LENGTH_TARGET_STR

        # Start from current TCP : do not hardcode frog/tip
        current_tcp = np.array(self.rtde_r.getActualTCPPose())

        if note['bow_dir'] == 'down':
            end = current_tcp + bow_fraction * (TIP_TARGET_STR - FROG_TARGET_STR)
        else:
            end = current_tcp + bow_fraction * (FROG_TARGET_STR - TIP_TARGET_STR)

        # Clip to frog-tip range
        for i in range(3):
            end[i] = np.clip(
                end[i],
                min(FROG_TARGET_STR[i], TIP_TARGET_STR[i]),
                max(FROG_TARGET_STR[i], TIP_TARGET_STR[i])
            )

        self._trajectory = [
            current_tcp + (i / max(n_steps - 1, 1)) * (end - current_tcp)
            for i in range(n_steps)
        ]
        self._n_steps_this_note = n_steps
        self._current_force     = force
        self._current_note      = note

        print(f"  Note {idx+1:3d}/{len(self.note_events)} | "
              f"dyn={note['dynamic']:2s}  vel={note['velocity']:3d}  "
              f"dur={note['duration']:.2f}s  "
              f"force={force:.2f}N  speed={speed:.3f}m/s  "
              f"dir={note['bow_dir']}  steps={n_steps}")

    def _stop_force_mode(self):
        pass  # force mode disabled : using z-offset position control


# -- Test function --------------------------

def run_baseline_test(midi_path: str):
    """
    Play a MIDI file using the baseline controller.

    """
    events = parse_midi_a_string(midi_path)
    ctrl   = TargetStringBaselineController(events)

    try:
        ctrl.reset()

        input("\nPress ENTER to begin playback (Ctrl+C to abort)...")
        ctrl.state_logger.start()

        step = 0
        while not ctrl.is_complete():
            baseline = ctrl.get_baseline_action()
            if baseline is None:
                break

            tcp   = baseline['tcp_target']
            force = baseline['force_nominal']

            ctrl.execute_timestep(tcp, force)
            state = ctrl.get_full_state()

            if step % 20 == 0:
                print(f"  step={step:4d} | "
                      f"force_cmd={force:.2f}N  "
                      f"fz={state['measured_force']:+.3f}N  "
                      f"contact_mag={state['contact_force_magnitude']:.3f}N | "
                      f"bow={state['bow_position']:.2f} | "
                      f"dyn={baseline['dynamic']}  "
                      f"dir={baseline['bow_direction']}")

            step += 1
            time.sleep(TIMESTEP_SEC)

        print(f"\nComplete. {step} steps executed.")

        ctrl.state_logger.stop()
        summary = ctrl.state_logger.get_summary()
        if summary:
            print(f"\nSession summary:")
            print(f"  Speed mean/max:    {summary['speed_mean']:.3f} / "
                  f"{summary['speed_max']:.3f} m/s  "
                  f"(p10={summary['speed_p10']:.3f}  p90={summary['speed_p90']:.3f})")
            print(f"  Torque contact:    {summary['torque_contact_est_N']:.3f} N est  "
                  f"(|τ|={summary['torque_mag_mean']:.4f} ± "
                  f"{summary['torque_mag_std']:.4f} N·m)")
            print(f"  Bow range:         {summary['bow_pos_start']:.2f} -> "
                  f"{summary['bow_pos_end']:.2f}")
            print(f"  Samples logged:    {summary['n_samples']}")

    except KeyboardInterrupt:
        print("\nAborted.")
        ctrl.state_logger.stop()
    finally:
        ctrl.close()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python baseline_controller.py <midi_file>")
        print("Example: python baseline_controller.py ../MIDI-Files/twinkle_twinkle-open.mid")
    else:
        run_baseline_test(sys.argv[1])