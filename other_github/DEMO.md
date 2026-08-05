# Demo — Audio → Next Force

A single-shot demo that takes a WAV file and prints the next force command
the PPO policy would issue.

## Run it

```bash
# With the trained model (recommended)
python demo.py \
    --audio "../CM/Shared_sound_model/data/Test_1_A3.wav" \
    --model checkpoints/ppo_cello_A_final.pt \
    --current-force 3.0

# Different current force to see how PPO reacts
python demo.py --audio path/to.wav --model checkpoints/ppo_cello_A_final.pt --current-force 6.0

# No trained model (fresh random weights) — pipeline still runs
python demo.py --audio path/to.wav
```

## Output sections

1. **Audio Input** — file metadata, duration, RMS amplitude
2. **Feature Extraction** — 6 librosa-based features + pitch (MIDI)
3. **Sound Classifier** — DeepMLP logits, softmax probs, Good/Bad label
4. **RL State** — 16-dim observation (most fields are stubs without a real robot)
5. **PPO Policy** — actor output, scaled delta, final force command

Final line:
```
RESULT: next force command = X.XX N
```

## Recording the video

For a clean video, run the demo on **three** different inputs and let the
viewer compare:

```bash
# 1. Audio classified Good (or close to Good), force at optimum
python demo.py --audio "../CM/Shared_sound_model/data/Test_1_A3.wav" --model checkpoints/ppo_cello_A_final.pt --current-force 3.5

# 2. Audio classified Bad, force too high
python demo.py --audio "../CM/Shared_sound_model/data/Test_2_A3.wav" --model checkpoints/ppo_cello_A_final.pt --current-force 6.0

# 3. Force too low — see if PPO recommends pushing up
python demo.py --audio "../CM/Shared_sound_model/data/note_007_A3.wav" --model checkpoints/ppo_cello_A_final.pt --current-force 1.0
```

Talk through each section as it scrolls past:
- "Here we load 1 second of audio"
- "We extract 7 features that summarise the cello sound"
- "The DeepMLP classifies it as Good or Bad"
- "The PPO policy reads the state, including current force"
- "It outputs the next force command"

## Known caveats (be ready to explain if asked)

- **librosa ≠ Essentia**: We re-implemented the 6 features using librosa
  because Essentia doesn't install on Windows. Numerical values differ
  slightly so predictions may disagree with the original notebook on some
  files. The team will retrain the classifier on librosa features later.
- **PPO trained on MockClassifier**: The saved model in `checkpoints/` was
  trained on the mock classifier (analytic Gaussian around 3.5N). It hasn't
  seen audio-driven rewards yet. Once we plug `RealSoundClassifier` into the
  training loop, behavior will adapt.
- **No real robot in the loop**: TCP pose and F/T sensor readings are
  stubbed to zero. PPO's responsiveness will be much stronger once
  real-robot observations feed in.
