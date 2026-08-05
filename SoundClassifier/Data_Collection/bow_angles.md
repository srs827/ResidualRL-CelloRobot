# How to Fill in the 3 Bow Angles from Polyscope

The unified `recording_script.py` now uses exactly **3 hand-calibrated bow angles** instead of interpolating across a continuum. This is safer (no calibration errors from interpolation) and lets RL explore the full 3D space between them.

## The 3 Angles

The script expects three poses:

1. **BOTTOM_ANGLE** — one extreme (e.g., bow tilted/yawed one way)
2. **REFERENCE_ANGLE** — the middle/neutral (the same as the zone experiment)
3. **TOP_ANGLE** — the other extreme (opposite tilt/yaw)

Each angle is defined by two 6-vector poses:
- `*_ANGLE_FROG`: the frog end of the bow at that angle
- `*_ANGLE_TIP`: the tip end of the bow at that angle

## Getting the Poses from Polyscope

### Step 1: Jog to BOTTOM_ANGLE

1. Open **Polyscope** (the UR teach pendant interface)
2. Go to **Program** → **Move** (or use the **Jog** tab)
3. Manually jog the robot's wrist to the first extreme orientation (e.g., bow tilted downward)
   - Use the **pendant** or **keyboard** to adjust rx, ry, rz until the bow is at the BOTTOM extreme
   - Keep the general bow geometry the same (still pointing along the A string)
4. Once positioned, **read the TCP position**:
   - In Polyscope, look for **Robot** → **TCP Position** or **Actual TCP Pose**
   - You'll see a 6-vector like: `[0.525, 0.351, 0.215, -1.370, -2.336, 1.327]`
   - **Copy this value exactly**

5. **Split into frog and tip**:
   - The TCP position at the frog is approximately:
     ```
     FROG_approx = BOTTOM_ANGLE_FROG_current  (where you start)
     ```
   - Actually, the 6-vector you see is the current TCP, which is somewhere along the stroke
   - You need TWO poses: one at the **frog end** and one at the **tip end** of the same angle
   
   **Better approach:**
   - Jog the robot so the bow touches the string at the **frog**
   - Read and copy the TCP pose → this is `BOTTOM_ANGLE_FROG`
   - Then jog the robot so the bow touches the string at the **tip**
   - Read and copy the TCP pose → this is `BOTTOM_ANGLE_TIP`
   - **Keep the wrist orientation (rx, ry, rz) the same** — only the x, y, z change as you move frog→tip

### Step 2: Jog to REFERENCE_ANGLE (middle)

1. This should be the **same orientation as the original zone experiment**
   - If you haven't changed it from the script, it's already filled in (equals `FROG_TARGET_STR` and `TIP_TARGET_STR`)
   - Or repeat the frog/tip process with the neutral/middle wrist orientation

### Step 3: Jog to TOP_ANGLE

1. Jog to the **opposite extreme** (e.g., bow tilted upward)
2. Again, jog to frog and tip separately, read the TCP poses
3. Copy `TOP_ANGLE_FROG` and `TOP_ANGLE_TIP`

## Updating the Script

Once you have the three pairs of poses, edit `recording_script.py`:

```python
BOTTOM_ANGLE_FROG = np.array([...paste frog TCP from bottom orientation...])
BOTTOM_ANGLE_TIP  = np.array([...paste tip TCP from bottom orientation...])

REFERENCE_ANGLE_FROG = np.array([...paste frog TCP from neutral...])
REFERENCE_ANGLE_TIP  = np.array([...paste tip TCP from neutral...])

TOP_ANGLE_FROG = np.array([...paste frog TCP from top orientation...])
TOP_ANGLE_TIP  = np.array([...paste tip TCP from top orientation...])
```

Make sure each is a numpy array with 6 floats: `[x, y, z, rx, ry, rz]`

## Example

If Polyscope shows:
```
Bottom angle — Frog end:     [0.3366, 0.7733, 0.1039, -1.4698, -2.3362, 1.2270]
Bottom angle — Tip end:      [0.5252, 0.3510, 0.2148, -1.4698, -2.3362, 1.2270]
Reference — Frog end:        [0.3366, 0.7733, 0.1039, -1.3698, -2.3362, 1.3270]
Reference — Tip end:         [0.5252, 0.3510, 0.2148, -1.3698, -2.3362, 1.3270]
Top angle — Frog end:        [0.3366, 0.7733, 0.1039, -1.2698, -2.3362, 1.4270]
Top angle — Tip end:         [0.5252, 0.3510, 0.2148, -1.2698, -2.3362, 1.4270]
```

Then edit the script:

```python
BOTTOM_ANGLE_FROG = np.array([0.3366, 0.7733, 0.1039, -1.4698, -2.3362, 1.2270])
BOTTOM_ANGLE_TIP  = np.array([0.5252, 0.3510, 0.2148, -1.4698, -2.3362, 1.2270])

REFERENCE_ANGLE_FROG = np.array([0.3366, 0.7733, 0.1039, -1.3698, -2.3362, 1.3270])
REFERENCE_ANGLE_TIP  = np.array([0.5252, 0.3510, 0.2148, -1.3698, -2.3362, 1.3270])

TOP_ANGLE_FROG = np.array([0.3366, 0.7733, 0.1039, -1.2698, -2.3362, 1.4270])
TOP_ANGLE_TIP  = np.array([0.5252, 0.3510, 0.2148, -1.2698, -2.3362, 1.4270])
```

## Running the Experiment

Once the 3 angles are filled in:

```bash
python recording_script.py
```

Set `EXPERIMENT_MODE = 'tilt_yaw'` (or `'both'` to run zone first, then tilt_yaw).

The script will record:
- **3 angles** × **5 depths** × **5 speeds** = 75 conditions
- Each condition recorded 5 times (default `N_REPEATS`)
- Total: **375 recordings** per tilt_yaw session
- Plus **4 bad conditions** at the reference angle for robustness

## Validation

After each stroke, the metadata includes:
- `commanded.angle_name` — which of the 3 angles (bottom/reference/top)
- `commanded.angle_t` — position on the continuum (0.0, 0.5, 1.0 for our 3 angles)
- `measured.bow_pos_start/end` — actual bow position during the stroke
- `measured.torque_contact_est_N` — contact force estimate (more reliable than |F|)

The session log also records all 3 angle poses for reference.

## Why This is Better

1. **No interpolation error** — the 3 angles are directly from hardware
2. **RL can learn the space** — training can find good angles between bottom/reference/top
3. **Safe** — you control the extremes, not a mathematical formula
4. **Reproducible** — the 3 fixed angles anchor the dataset