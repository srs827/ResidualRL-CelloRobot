"""
acceptance_mount.py — bow-mount acceptance test (post silicone-tape fix, 2026-06-09).

Static press-release sweeps at TWO bow positions. Built ENTIRELY on the proven
recording_grid path (presses along -RETRACT_DIR_BASE via apply_force_offset,
zeroFtSensor tare, clamp_reach, perpendicular reset_to / vertical-first retract).
Does NOT touch hardware_baseline (known depth-reference bugs there).

Per position (frog-quarter frac=0.275 and mid-bow frac=0.50), per run:
  re-tare in free air -> descend to light contact -> DOWN sweep 0->3.5mm
  (0.25mm steps) -> UP sweep back -> retract.

Verdict per position (hard criteria -> PASS):
  dead zone   : x-intercept of the force-depth fit on the responding segment
                (a clean spring fits through ~0; backlash shifts the intercept
                by the gap size — slope/step independent)      PASS <= 0.3 mm
  hysteresis  : mean |F_down - F_up| on the RESPONDING grid    PASS <  0.15 N
  monotonic   : depth-force corr on the responding segment     PASS >= 0.90
Soft (reported, not gating): slope vs the bow-physics band (0.25-0.45 N/mm
frog-quarter, 0.15-0.30 mid-bow), slope spread across runs < 20%, F at depth 0
(preload flag). A mount with NO force response at all = automatic FAIL.

Live safety: abort + retract the moment |F| > 5.0 N; an abort ends that
position (no further pressing there). Results are saved INCREMENTALLY after
every position and on any exception.

Run from the repo root (robot in Remote mode, HAND ON E-STOP; no audio needed):
  .venv-rtde/bin/python acceptance_mount.py             # 2 positions x 3 runs
  .venv-rtde/bin/python acceptance_mount.py --quick     # 1 run per position
  .venv-rtde/bin/python acceptance_mount.py --frac 0.5  # single position

Author: Zixian + Claude, 2026-06-09 (rev 2 after adversarial review: numpy-safe
JSON, x-intercept dead zone, live-region hysteresis, per-run tare, incremental
save, abort ends position, --frac validation, first-motion retract).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recording_grid as R                                    # noqa: E402

STEP_M = 0.00025          # 0.25 mm per step
MAX_DEPTH_M = 0.0035      # sweep ceiling RELATIVE to measured light contact
MAX_PRESS_LIMIT = 0.004   # absolute ceiling vs the nominal waypoint (recording_grid)
MAX_FORCE_N = 5.0         # live abort cap (bow structural margin near tip is 3-6 N)
SETTLE_S = 0.4            # per-step settle (tape is viscoelastic; let it relax)
ROBOT_IP = "192.168.1.100"
POSITIONS = {"frog_quarter": 0.275, "mid_bow": 0.50}
OUT_DIR = os.path.join(HERE, "mount_acceptance")


def slope_band(frac):
    """Expected depth-to-force gain band (N/mm) from verified bow physics."""
    return (0.25, 0.45) if frac < 0.40 else (0.15, 0.30)


def set_speed_slider_full(ip=ROBOT_IP):
    """Inline (NOT imported from pre_experiment_grid — importing that module
    mutates recording_grid globals as a side effect)."""
    import rtde_io
    try:
        rtde_io.RTDEIOInterface(ip).setSpeedSlider(1.0)
        print("  speed slider -> 100%")
    except Exception as e:
        print(f"  (speed slider not set: {e}) — check the pendant slider manually!")


def safety_ok(ctrl):
    """True only if the robot is in Normal safety mode and Running (lesson from
    2026-06-09 first run: in protective stop every command fails SILENTLY and
    the receive buffer freezes — audit R3/R4/R18)."""
    try:
        if ctrl.rtde_r.isProtectiveStopped():
            return False
        if int(ctrl.rtde_r.getSafetyMode()) != 1:      # 1 = NORMAL
            return False
        if int(ctrl.rtde_r.getRobotMode()) != 7:       # 7 = RUNNING
            return False
    except Exception:
        return False   # FAIL CLOSED (parity with hardware_baseline): a dead receive
                       # interface IS the protective-stop signature, never "safe"
    return True


def require_safe(ctrl, what):
    if not safety_ok(ctrl):
        raise RuntimeError(
            f"ROBOT NOT NORMAL/RUNNING ({what}). Pendant: dismiss the safety popup, "
            "Initialize -> Enable Robot (brakes released, status Running), set REMOTE "
            "mode, then rerun. Data collected so far is already saved.")


def f_now(ctrl):
    """Tared contact force: magnitude + press-axis component (string pushes the
    tool AWAY from the string, i.e. along +RETRACT_DIR, so pressing reads +)."""
    f3 = np.array(ctrl.rtde_r.getActualTCPForce()[:3])
    return float(np.linalg.norm(f3)), float(np.dot(f3, R.RETRACT_DIR_BASE))


def noise_floor(ctrl, seconds=1.0):
    vals = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        vals.append(f_now(ctrl)[0])
        time.sleep(0.02)
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    if sd == 0.0:
        raise RuntimeError(
            f"Force stream FROZEN ({len(vals)} identical samples = stale RTDE receive "
            "buffer; robot likely protective-stopped). Recover on the pendant, rerun.")
    return mu, sd


SEARCH_START_M = -0.012   # begin 12mm above the (stale) nominal waypoint


def find_contact(ctrl, waypoint, clearance_mu):
    """Descend from SEARCH_START toward the string and return the offset (m,
    negative = above the nominal waypoint) of TRUE light contact (= depth 0).

    Onset detection is DIRECTIONAL and locally-baselined: contact force points
    along the press axis (F_axis/|F| was ~0.95 in real sweeps), while cable drag
    and tare drift do not — so we trigger on the press-axis INCREMENT over a
    baseline sampled at the search start, immune to the |F| offset that caused
    the 0.97N false 'geometry changed' abort on 2026-06-10."""
    # local baseline at the start hover (caller already moved us there)
    vals = []
    t0 = time.time()
    while time.time() - t0 < 0.6:
        vals.append(f_now(ctrl)[1])
        time.sleep(0.02)
    base_a, base_sd = float(np.mean(vals)), float(np.std(vals))
    m0 = f_now(ctrl)[0]
    if m0 - clearance_mu > 2.0:
        R.retract(ctrl)
        raise RuntimeError(f"|F| at the search-start hover is {m0:.2f}N (clearance was "
                           f"{clearance_mu:.2f}N) — likely ALREADY in hard contact 12mm above "
                           "the waypoint; re-teach the string waypoints first.")
    onset_delta = max(0.25, 5 * base_sd)
    print(f"  contact search: local baseline F_axis={base_a:+.3f}±{base_sd:.3f}N, "
          f"onset = ΔF_axis > {onset_delta:.2f}N, from {SEARCH_START_M*1000:.0f}mm above nominal...")
    d, onset, prev_hit = SEARCH_START_M, None, False
    while d <= MAX_PRESS_LIMIT + 1e-9:
        pose = R.clamp_reach(R.apply_force_offset(waypoint, float(d)))
        ok = ctrl.rtde_c.moveL(pose.tolist(), ctrl.SERVO_VEL, ctrl.SERVO_ACCEL)
        if ok is False or not safety_ok(ctrl):
            raise RuntimeError("moveL failed / robot left Normal state during contact search.")
        time.sleep(SETTLE_S)
        m, a = f_now(ctrl)
        da = a - base_a
        print(f"    offset={d*1000:+6.2f}mm  |F|={m:5.3f}N  F_axis={a:+5.3f}N  dA={da:+5.3f}N")
        if m > MAX_FORCE_N:
            R.retract(ctrl)
            raise RuntimeError("force cap hit during contact search — geometry is way off, re-teach waypoints.")
        if da > onset_delta:
            if prev_hit and onset is not None:
                return onset                      # two consecutive hits -> onset confirmed
            prev_hit, onset = True, float(d - STEP_M)   # back off one step = light contact
        else:
            prev_hit, onset = False, None
        d += STEP_M
    raise RuntimeError("no contact found over the whole search range — string missing?")


def sweep(ctrl, waypoint, direction, base=0.0):
    """One pass over the depth grid, offset by `base` (the measured light-contact
    offset from find_contact; depths are recorded RELATIVE to that contact).
    +1 = press in (0->max), -1 = release (starts ONE step below max so the shared
    endpoint doesn't fake zero hysteresis). Returns plain-float lists."""
    grid = np.arange(0.0, MAX_DEPTH_M + 1e-9, STEP_M)
    if direction < 0:
        grid = grid[::-1][1:]
    grid = base + grid
    grid = grid[grid <= MAX_PRESS_LIMIT]          # never exceed the absolute ceiling
    depths, fmag, faxis, aborted = [], [], [], False
    for d in grid:
        pose = R.clamp_reach(R.apply_force_offset(waypoint, float(d)))
        ok = ctrl.rtde_c.moveL(pose.tolist(), ctrl.SERVO_VEL, ctrl.SERVO_ACCEL)
        if ok is False or not safety_ok(ctrl):
            raise RuntimeError("moveL failed / robot left Normal state mid-sweep "
                               "(protective stop?). Positions finished so far are saved.")
        time.sleep(SETTLE_S)
        m, a = f_now(ctrl)
        rel = float(d - base)                     # depth relative to TRUE light contact
        depths.append(rel); fmag.append(float(m)); faxis.append(float(a))
        print(f"    depth={rel*1000:5.2f}mm  |F|={m:5.3f}N  F_axis={a:+5.3f}N")
        if m > MAX_FORCE_N:
            print(f"    !! |F| > {MAX_FORCE_N} N — ABORT, retracting; this position ends here.")
            R.retract(ctrl)
            aborted = True
            break
    return depths, fmag, faxis, aborted


def analyze(frac, noise_mu, noise_sd, runs):
    """Acceptance criteria for one position. All outputs are plain Python types."""
    thresh = max(0.10, noise_mu + 5 * noise_sd)
    lo, hi = slope_band(frac)
    out = {"threshold_N": float(thresh), "slope_band_N_per_mm": [lo, hi], "runs": []}
    slopes = []
    clean = [r for r in runs if not r.get("aborted")]
    for r in clean:
        d_dn = np.array(r["down"]["depth"]); f_dn = np.array(r["down"]["fmag"])
        d_up = np.array(r["up"]["depth"]);   f_up = np.array(r["up"]["fmag"])
        a = {"f_at_zero_N": float(f_dn[0]) if len(f_dn) else None}
        live = f_dn > thresh
        if live.sum() >= 3:
            dd, ff = d_dn[live], f_dn[live]
            slope_m, intercept = np.polyfit(dd, ff, 1)          # F = slope_m*d + b
            slope = float(slope_m) / 1000.0                      # N/m -> N/mm
            # dead zone = x-intercept of the responding-segment fit, clipped >= 0:
            # a clean spring fits through ~0; backlash shifts it by the gap size.
            dz_m = float(np.clip(-intercept / max(slope_m, 1e-9), 0.0, MAX_DEPTH_M))
            a["dead_zone_mm"] = float(dz_m * 1000)
            a["slope_N_per_mm"] = slope
            a["corr"] = float(np.corrcoef(dd, ff)[0, 1]) if float(np.std(ff)) > 0 else 0.0
            a["preloaded"] = bool(a["f_at_zero_N"] is not None and a["f_at_zero_N"] > thresh)
            slopes.append(slope)
            # hysteresis on the RESPONDING region only (pre-contact pairs are
            # trivially ~equal and would dilute the mean)
            live_keys = set(np.round(d_dn[live], 6).tolist())
            up_map = dict(zip(np.round(d_up, 6).tolist(), f_up.tolist()))
            pairs = [(float(f), float(up_map[k]))
                     for k, f in zip(np.round(d_dn, 6).tolist(), f_dn.tolist())
                     if k in up_map and k in live_keys]
            if pairs:
                diffs = [abs(x - y) for x, y in pairs]
                a["hysteresis_mean_N"] = float(np.mean(diffs))
                a["hysteresis_max_N"] = float(np.max(diffs))
            # press-AXIS hysteresis = the real backlash fingerprint (|F| gets
            # contaminated by lateral rosin stiction during release — seen 06-09)
            fa_dn = np.array(r["down"]["faxis"]); fa_up = np.array(r["up"]["faxis"])
            up_map_a = dict(zip(np.round(d_up, 6).tolist(), fa_up.tolist()))
            pairs_a = [(float(f), float(up_map_a[k]))
                       for k, f in zip(np.round(d_dn, 6).tolist(), fa_dn.tolist())
                       if k in up_map_a and k in live_keys]
            if pairs_a:
                diffs_a = [abs(x - y) for x, y in pairs_a]
                a["hysteresis_axis_mean_N"] = float(np.mean(diffs_a))
        else:
            a["no_response"] = True
        a["force_range_N"] = float(f_dn.max() - f_dn.min()) if len(f_dn) else 0.0
        out["runs"].append(a)
    if len(slopes) >= 2:
        out["slope_spread_pct"] = float(100 * (max(slopes) - min(slopes))
                                        / max(abs(float(np.mean(slopes))), 1e-9))
    out["n_aborted_runs"] = int(sum(1 for r in runs if r.get("aborted")))
    dzs = [r["dead_zone_mm"] for r in out["runs"] if "dead_zone_mm" in r]
    hys = [h for h in (r.get("hysteresis_axis_mean_N", r.get("hysteresis_mean_N"))
                       for r in out["runs"]) if h is not None]
    cor = [r["corr"] for r in out["runs"] if "corr" in r]
    responded = bool(dzs)
    out["responded"] = responded
    out["pass_dead_zone"] = bool(responded and max(dzs) <= 0.30)
    out["pass_hysteresis"] = bool(hys) and bool(max(hys) < 0.15)
    out["pass_monotonic"] = bool(cor) and bool(min(cor) >= 0.90)
    out["pass_repeatable"] = (bool(out["slope_spread_pct"] < 20.0)
                              if "slope_spread_pct" in out else None)
    out["slope_in_expected_band"] = (bool(lo <= float(np.mean(slopes)) <= hi)
                                     if slopes else None)
    # no force response at all = the original broken-mount failure -> hard FAIL
    out["PASS"] = bool(responded and out["pass_dead_zone"]
                       and out["pass_hysteresis"] and out["pass_monotonic"])
    return out


def save(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"acceptance_{results['timestamp']}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 run per position instead of 3")
    ap.add_argument("--frac", type=float, default=None, help="test a single bow fraction")
    ap.add_argument("--search-mm", type=float, default=12.0,
                    help="contact-search start height above the nominal waypoint (mm); "
                         "deepen to ~18 for high fracs where the string line plateaus")
    args = ap.parse_args()
    global SEARCH_START_M
    SEARCH_START_M = -abs(args.search_mm) / 1000.0
    n_runs = 1 if args.quick else 3
    if args.frac is not None:
        if not (0.20 <= args.frac <= 0.90):
            ap.error("--frac must be in [0.20, 0.90] (frog end is reach/singularity-unsafe)")
        reach = float(np.linalg.norm(R.bow_pose_full(args.frac)[:3]))
        if reach > R.MAX_SAFE_REACH - 0.005:
            ap.error(f"--frac {args.frac}: waypoint reach {reach:.3f} too close to "
                     f"MAX_SAFE_REACH {R.MAX_SAFE_REACH} (would be silently rescaled)")
        positions = {"custom": args.frac}
    else:
        positions = dict(POSITIONS)

    print(">>> Mount acceptance test. Robot in REMOTE mode. HAND ON THE E-STOP. <<<")
    ctrl = R.TargetStringBaselineController([])
    set_speed_slider_full()
    results = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
               "step_mm": STEP_M * 1000, "max_depth_mm": MAX_DEPTH_M * 1000,
               "max_force_N": MAX_FORCE_N, "settle_s": SETTLE_S, "positions": {}}
    path = None
    try:
        require_safe(ctrl, "preflight — robot must be Normal/Running BEFORE any motion")
        # First motion depends on where the arm is. R.retract() (+3cm up, then 60mm
        # OUTWARD) is proven from ON-STRING poses only — from a high parked pose it
        # pushes an already-extended arm past the envelope (caused the 06-09 C204A3 /
        # C306A3 faults). If we're parked high, skip it: the tare's own first move
        # (clamp_reach'd clearance pose) approaches inward/downward and is safe.
        tcp0 = np.array(ctrl.rtde_r.getActualTCPPose())
        near_string = tcp0[2] < 0.30
        print(f"  start pose: z={tcp0[2]:.3f} -> {'near string: retract first' if near_string else 'parked high: going directly to the tare clearance pose'}")
        input("\nENTER to start (confirm the bow is not tangled in anything)...")
        if near_string:
            R.retract(ctrl)
            require_safe(ctrl, "after initial retract")
        first = True
        for name, frac in positions.items():
            input(f"\nENTER: position '{name}' (frac={frac:.3f}) — {n_runs} run(s), "
                  f"re-tared each run...")
            waypoint = R.bow_pose_full(frac)
            runs = []
            mu = sd = None
            for k in range(n_runs):
                print(f"  run {k + 1}/{n_runs}: tare in free air, then descend...")
                f0 = R.zero_ft_in_free_air(ctrl, at_frac=frac, set_payload=first)
                first = False
                require_safe(ctrl, "after tare moves")
                if f0 is not None and f0 > 1.0:
                    raise RuntimeError(
                        f"TARE FAILED: free-air |F|={f0:.2f}N after zeroFtSensor (should be "
                        "~0). zeroFtSensor returned False or the payload is wrong — do not "
                        "trust any force numbers; fix robot state first.")
                mu, sd = noise_floor(ctrl)
                print(f"  noise floor: {mu:.3f} ± {sd:.3f} N")
                # hover 12mm ABOVE the (stale) nominal waypoint — NEVER descend onto it
                hover = R.clamp_reach(R.apply_force_offset(waypoint, SEARCH_START_M))
                R.reset_to(ctrl, hover)
                require_safe(ctrl, "after descent to hover")
                time.sleep(0.4)
                contact = find_contact(ctrl, waypoint, mu)
                print(f"  TRUE light contact at {contact*1000:+.2f}mm vs nominal waypoint")
                print(f"  >>> CONTACT_ANCHORS entry — paste EXACTLY (meters, KEEP the sign): "
                      f"({frac:.3f}, {contact:+.5f})")
                print("  DOWN sweep (depths relative to true contact):")
                dd, fm, fa, ab1 = sweep(ctrl, waypoint, +1, base=contact)
                ab2 = False
                du, fmu_, fau = [], [], []
                if not ab1:
                    print("  UP sweep:")
                    du, fmu_, fau, ab2 = sweep(ctrl, waypoint, -1, base=contact)
                runs.append({"down": {"depth": dd, "fmag": fm, "faxis": fa},
                             "up": {"depth": du, "fmag": fmu_, "faxis": fau},
                             "contact_offset_mm": float(contact * 1000),
                             "aborted": bool(ab1 or ab2)})
                R.retract(ctrl)
                if ab1 or ab2:
                    print("  abort -> ending this position (no further pressing here).")
                    break
            ana = analyze(frac, mu or 0.0, sd or 0.0, runs)
            results["positions"][name] = {"frac": frac, "noise_mu_N": mu, "noise_sd_N": sd,
                                          "runs": runs, "analysis": ana}
            path = save(results)   # incremental: survive anything that happens later
            print(f"\n  === {name} verdict ===   (saved -> {path})")
            for key in ("responded", "pass_dead_zone", "pass_hysteresis", "pass_monotonic",
                        "pass_repeatable", "slope_in_expected_band", "PASS"):
                print(f"    {key}: {ana.get(key)}")
    finally:
        if results["positions"]:
            path = save(results)
        try:
            R.retract(ctrl)
        except Exception:
            pass
        ctrl.close()

    if path:
        print(f"\nsaved -> {path}")
    done = results["positions"]
    if done:
        overall = all(p["analysis"]["PASS"] for p in done.values())
        print(f"\n=== OVERALL: {'PASS — mount backlash fixed' if overall else 'NOT PASSING — see flags above'} ===")


if __name__ == "__main__":
    main()
