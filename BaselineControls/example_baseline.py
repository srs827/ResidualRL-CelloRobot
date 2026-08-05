"""

Drop-in placeholder for BaselineController.

Usage:
    from example_baseline import ExBaselineController as BaselineController

"""

import numpy as np
import time

# fake values
BOW_POSES = {
    'A': {
        'frog': np.array([.336637615375, .773335607743, .103937252349,
                          -1.369835518384, -2.336199267621,  1.326965437172]),
        'tip':  np.array([.525205911288, .350983193771, .214779688012,
                          -1.369835518377, -2.336199267606,  1.326965437192]),
    },
    'D': {
        'frog': np.array([.3028, .7498, .1173, -1.6641, -2.0843, 1.0380]),
        'tip':  np.array([.3404, .2802, .1763, -1.6146, -2.0448, 1.0423]),
    },
    'G': {
        'frog': np.array([.2812, .6817, .1047, -1.8122, -1.9402, 0.4937]),
        'tip':  np.array([.1620, .2013, .0594, -1.9298, -1.9313, 0.5551]),
    },
    'C': {
        'frog': np.array([.2567, .6101, .0626, -1.7432, -1.5245, 0.1638]),
        'tip':  np.array([.0798, .2852, -.0867, -1.8196, -1.6583, 0.1809]),
    },
}

# fake force vals
BASELINE_FORCE = {
    'A': 2.8,
    'D': 3.1,
    'G': 3.5,
    'C': 4.0,
}

# Residual action bounds — agreed interface contract
MAX_DELTA_XY  = 0.003   # m +/- 3mm lateral
MAX_DELTA_Z   = 0.002   # m +/- 2mm into/away from string
MAX_DELTA_F   = 0.3     # N  +/- 0.3N force

# Strings in order 
STRING_ORDER = ['C', 'G', 'D', 'A']


class ExBaselineController:
    """
    Example baseline controller for RL development.

    """

    def __init__(
        self,
        note_sequence,          # from parse_midi() in robot_runner.py
        n_steps_per_stroke = 150,
        noise_std          = 0.0001,   # small TCP noise to simulate real robot
    ):
        self.note_sequence     = note_sequence
        self.n_steps           = n_steps_per_stroke
        self.noise_std         = noise_std

        # State
        self.current_note_idx  = 0
        self.current_step      = 0
        self.bow_direction     = True    # True = down bow (frog -> tip)
        self._trajectory       = []
        self._current_string   = None

        # Simulated sensor state
        self._sim_tcp          = np.zeros(6)
        self._sim_force        = 0.0


    def reset(self, string: str = None):
        """
        Reset to start of episode.
        string: override starting string (None = use first note in sequence)
        """
        self.current_note_idx  = 0
        self.current_step      = 0
        self.bow_direction     = True

        if string is not None:
            # Find first note matching this string
            for i, note in enumerate(self.note_sequence):
                if note['string'] == string and '-' not in note['string']:
                    self.current_note_idx = i
                    break

        self._build_trajectory()

    def get_baseline_action(self) -> dict:
        """
        Returns baseline target for current timestep.
        Call every 50ms during RL execution.

        Returns None when sequence is complete.
        """
        if self.current_note_idx >= len(self.note_sequence):
            return None

        note   = self.note_sequence[self.current_note_idx]
        string = note['string']

        # Handle transition
        if '-' in string:
            return self._transition_action(string)

        # End of stroke — advance to next note
        if self.current_step >= self.n_steps:
            self.current_note_idx += 1
            self.current_step      = 0
            self.bow_direction     = not self.bow_direction

            if self.current_note_idx >= len(self.note_sequence):
                return None

            self._build_trajectory()
            note   = self.note_sequence[self.current_note_idx]
            string = note['string']

            if '-' in string:
                return self._transition_action(string)

        # Normal stroke step
        tcp   = self._trajectory[self.current_step].copy()
        force = BASELINE_FORCE.get(string, 3.0)

        # Simulate small robot noise
        tcp[:3] += np.random.normal(0, self.noise_std, 3)

        # Update simulated sensor state
        self._sim_tcp   = tcp
        self._sim_force = force + np.random.normal(0, 0.05)
        self._current_string = string

        bow_pos = self.current_step / self.n_steps
        self.current_step += 1

        return {
            'tcp_target':    tcp,
            'force_nominal': force,
            'bow_position':  float(bow_pos),
            'bow_direction': 'down' if self.bow_direction else 'up',
            'string':        string,
            'is_transition': False,
            'step':          self.current_step,
            'note_duration': float(note['duration']),
        }

    def get_score_context(self) -> dict:
        """Current MIDI score position for RL observation."""
        if self.current_note_idx >= len(self.note_sequence):
            return {
                'current_string':  None,
                'time_remaining':  0.0,
                'next_string':     None,
                'bow_position':    1.0,
                'is_transition':   False,
                'sequence_done':   True,
            }

        note     = self.note_sequence[self.current_note_idx]
        string   = note['string']
        bow_pos  = self.current_step / self.n_steps
        timestep_sec = 0.05

        next_note = (self.note_sequence[self.current_note_idx + 1]
                     if self.current_note_idx + 1 < len(self.note_sequence)
                     else None)

        return {
            'current_string':  string,
            'time_remaining':  float((1.0 - bow_pos) * note['duration']),
            'next_string':     next_note['string'] if next_note else None,
            'bow_position':    float(bow_pos),
            'bow_direction':   'down' if self.bow_direction else 'up',
            'is_transition':   '-' in string,
            'note_duration':   float(note['duration']),
            'sequence_done':   False,
        }

    def apply_residual(
        self,
        baseline_action: dict,
        residual: np.ndarray,
    ) -> tuple:
        """
        Apply residual correction to baseline action.

        residual: np.ndarray shape (3,) or (4,)
            [dx, dy, dz]          if shape (3,) — position only
            [dx, dy, dz, dforce]  if shape (4,) — position + force

        All values normalized [-1, 1].
        Returns: (tcp_target np.ndarray(6,), force float)
        """
        tcp   = baseline_action['tcp_target'].copy()
        force = baseline_action['force_nominal']

        # Position corrections — scaled and clipped
        tcp[0] += float(np.clip(residual[0] * MAX_DELTA_XY, -MAX_DELTA_XY, MAX_DELTA_XY))
        tcp[1] += float(np.clip(residual[1] * MAX_DELTA_XY, -MAX_DELTA_XY, MAX_DELTA_XY))
        tcp[2] += float(np.clip(residual[2] * MAX_DELTA_Z,  -MAX_DELTA_Z,  MAX_DELTA_Z))

        # Force correction (optional 4th action dimension)
        if len(residual) >= 4:
            force += float(np.clip(residual[3] * MAX_DELTA_F, -MAX_DELTA_F, MAX_DELTA_F))

        # Hard safety clamp
        force = float(np.clip(force, 0.3, 8.0))

        return tcp, force

    def get_simulated_state(self) -> dict:
        """
        Returns simulated robot state for RL observation.
        Real BaselineController returns actual RTDE readings.
        Same keys — RL observation builder uses this dict.
        """
        return {
            'tcp':           self._sim_tcp.copy(),
            'ft':            self._get_sim_ft(),
            'joints':        self._get_sim_joints(),
            'bow_speed':     self._get_sim_bow_speed(),
            'bow_position':  float(self.current_step / self.n_steps),
            'measured_force': float(self._sim_force),
            'bow_direction': 1.0 if self.bow_direction else -1.0,
        }

    def is_complete(self) -> bool:
        """True when MIDI sequence is finished."""
        return self.current_note_idx >= len(self.note_sequence)

    # ── Private helpers ───────────────────────────────────────────

    def _build_trajectory(self):
        """Interpolate TCP trajectory for current note."""
        if self.current_note_idx >= len(self.note_sequence):
            self._trajectory = []
            return

        note   = self.note_sequence[self.current_note_idx]
        string = note['string']

        if '-' in string:
            self._trajectory = []
            return

        poses = BOW_POSES.get(string, BOW_POSES['A'])

        if self.bow_direction:
            start = poses['frog']
            end   = poses['tip']
        else:
            start = poses['tip']
            end   = poses['frog']

        self._trajectory = [
            start + (i / (self.n_steps - 1)) * (end - start)
            for i in range(self.n_steps)
        ]

    def _transition_action(self, string_pair: str) -> dict:
        """Handle string crossing transition."""
        src, dst = string_pair.split('-')

        src_mid = (BOW_POSES[src]['frog'] + BOW_POSES[src]['tip']) / 2
        dst_mid = (BOW_POSES[dst]['frog'] + BOW_POSES[dst]['tip']) / 2

        n_transition = max(4, int(
            self._crossing_difficulty(src, dst) * 20
        ))

        t   = min(self.current_step / max(n_transition - 1, 1), 1.0)
        tcp = src_mid + t * (dst_mid - src_mid)

        self.current_step += 1

        if self.current_step >= n_transition:
            self.current_note_idx += 1
            self.current_step      = 0
            self._build_trajectory()

        self._sim_tcp   = tcp.copy()
        self._sim_force = 0.0   # no force during crossing

        return {
            'tcp_target':    tcp,
            'force_nominal': 0.0,
            'bow_position':  0.5,
            'bow_direction': 'down' if self.bow_direction else 'up',
            'string':        dst,
            'is_transition': True,
            'step':          self.current_step,
            'note_duration': 0.2,
        }

    def _crossing_difficulty(self, src: str, dst: str) -> float:
        """
        Returns relative crossing difficulty 0-1.
        Adjacent strings = 0.25, C to A = 1.0.
        Used to scale transition duration.
        """
        src_idx = STRING_ORDER.index(src) if src in STRING_ORDER else 0
        dst_idx = STRING_ORDER.index(dst) if dst in STRING_ORDER else 0
        return abs(src_idx - dst_idx) / (len(STRING_ORDER) - 1)

    def _get_sim_ft(self) -> np.ndarray:
        """Simulated F/T sensor — correlated with position."""
        ft = np.zeros(6)
        if self._current_string and not self.is_complete():
            ft[2] = self._sim_force    # fz = normal force on string
            ft[0] = np.random.normal(0, 0.1)   # fx = small lateral noise
            ft[3] = np.random.normal(0, 0.02)  # tx = small torque noise
        return ft

    def _get_sim_joints(self) -> np.ndarray:
        """Placeholder joint angles — not used in V1 observation."""
        return np.zeros(6)

    def _get_sim_bow_speed(self) -> float:
        """Approximate bow speed from trajectory step size."""
        if len(self._trajectory) < 2:
            return 0.0
        step     = min(self.current_step, len(self._trajectory) - 1)
        prev     = max(step - 1, 0)
        delta    = np.linalg.norm(
            self._trajectory[step][:3] - self._trajectory[prev][:3]
        )
        return float(delta / 0.05)   # delta_position / timestep