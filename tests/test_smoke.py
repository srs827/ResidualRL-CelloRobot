from pathlib import Path

import numpy as np

from rl.piece_env import MockExecutor, MockScorer, PieceResidualEnv


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mock_piece_rollout_completes():
    env = PieceResidualEnv(
        REPO_ROOT / "MIDI-Files" / "t1.mid",
        executor=MockExecutor(noise=0.0, rng=np.random.default_rng(0)),
        scorer=MockScorer(noise=0.0, rng=np.random.default_rng(0)),
    )

    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == env.observation_space.shape

        done = False
        steps = 0
        while not done:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation.shape == env.observation_space.shape
            assert np.isfinite(reward)
            done = terminated or truncated
            steps += 1
            assert steps < 100

        assert steps == len(env.plan)
        assert 0.0 <= info["mean_quality"] <= 1.0
    finally:
        env.close()
