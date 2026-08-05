"""
Example: full RL training loop using mock components.
When real components are ready, change the imports only.
"""

# ── Swap these imports when real components are ready ─────────────

from mock_baseline   import MockBaselineController as BaselineController
from mock_classifier import MockSoundClassifier    as SoundClassifier

# from baseline_controller import BaselineController   # ← real robot
# from classifier          import BowingQualityClassifier as SoundClassifier

# ── Everything below stays exactly the same ───────────────────────

from robot_runner import parse_midi
from reward      import CompleteCelloReward
from env         import SingleStringResidualEnv
from ppo         import PPOAgent

# Setup
note_sequence = parse_midi("MIDI-Files/twinkle_twinkle-open.mid")
baseline      = BaselineController(note_sequence)
classifier    = SoundClassifier()
reward_fn     = CompleteCelloReward(classifier_model=classifier)

env   = SingleStringResidualEnv(
    baseline      = baseline,
    reward_fn     = reward_fn,
    string        = 'A',
    max_steps     = 150,
)
agent = PPOAgent(env=env)

# Train — works identically with mock or real components
agent.learn(total_timesteps=50_000)