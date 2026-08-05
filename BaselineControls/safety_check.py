"""
preflight_check.py

Run this before baseline_controller.py.
Verifies waypoints, trajectory, and motion plan
without moving the robot.

"""

import numpy as np
import rtde_receive
import rtde_control

ROBOT_IP = "192.168.1.100"

# A string waypoints from a_full_bow.script
FROG_A = np.array([.336637615375,  .773335607743,  .103937252349,
                   -1.369835518384, -2.336199267621, 1.326965437172])
TIP_A  = np.array([.525205911288,  .350983193771,  .214779688012,
                   -1.369835518377, -2.336199267606, 1.326965437192])

BOW_LENGTH_A = float(np.linalg.norm(TIP_A[:3] - FROG_A[:3]))


def preflight(midi_path: str):
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

    print("\n" + "="*60)
    print("PRE-FLIGHT CHECK")
    print("="*60)

    all_pass = True

    # -- Check 1: Robot is reachable ------------------------------
    print("\n[1] Robot connection...")
    try:
        tcp = rtde_r.getActualTCPPose()
        print(f"    Connected. TCP: {[f'{v:.4f}' for v in tcp]}")
    except Exception as e:
        print(f"    Cannot connect: {e}")
        return False

    # -- Check 2: Current position vs frog waypoint ---------------
    print("\n[2] Current position vs FROG_A waypoint...")
    current_pos = np.array(tcp[:3])
    frog_pos    = FROG_A[:3]
    tip_pos     = TIP_A[:3]

    dist_to_frog = float(np.linalg.norm(current_pos - frog_pos))
    dist_to_tip  = float(np.linalg.norm(current_pos - tip_pos))

    print(f"    Current TCP:  {[f'{v:.4f}' for v in tcp[:3]]}")
    print(f"    FROG_A:       {[f'{v:.4f}' for v in frog_pos]}")
    print(f"    TIP_A:        {[f'{v:.4f}' for v in tip_pos]}")
    print(f"    Distance to frog: {dist_to_frog*100:.1f}cm")
    print(f"    Distance to tip:  {dist_to_tip*100:.1f}cm")
    print(f"    Bow length (frog→tip): {BOW_LENGTH_A*100:.1f}cm")

    if dist_to_frog < 0.05:
        print("    Robot is near frog position — ready to start")
    elif dist_to_frog < 0.15:
        print("    Robot is within 15cm of frog — acceptable but not ideal")
        print("      Manually jog to frog position before running")
    else:
        print(f"    Robot is {dist_to_frog*100:.1f}cm from frog")
        print("      This is far from the starting waypoint.")
        print("      Verify FROG_A matches actual frog position on your robot")
        print("      OR manually jog robot to frog before running")
        all_pass = False

    # -- Check 3: Orientation at frog vs stored waypoint -----------
    print("\n[3] Orientation check...")
    current_rot = np.array(tcp[3:])
    frog_rot    = FROG_A[3:]
    rot_error   = np.linalg.norm(current_rot - frog_rot)

    print(f"    Current rx,ry,rz: {[f'{v:.4f}' for v in current_rot]}")
    print(f"    FROG_A  rx,ry,rz: {[f'{v:.4f}' for v in frog_rot]}")
    print(f"    Rotation error:   {np.degrees(rot_error):.1f} degrees")

    if rot_error < 0.1:
        print("    Orientation matches frog waypoint")
    else:
        print(f"    Orientation differs by {np.degrees(rot_error):.1f}°")
        print("      If robot was moved manually this is expected")
        print("      The baseline will move to FROG_A on reset() — this is OK")

    # -- Check 4: MIDI file parses correctly ------------------------
    print(f"\n[4] MIDI file: {midi_path}")
    try:
        from baseline_controller import parse_midi_a_string
        events = parse_midi_a_string(midi_path)

        if not events:
            print("    No note events parsed — check MIDI file")
            all_pass = False
        else:
            durations = [e['duration'] for e in events]
            velocities = [e['velocity'] for e in events]
            bow_dirs = [e['bow_dir'] for e in events]

            print(f"    {len(events)} notes parsed")
            print(f"    Duration range: {min(durations):.2f}s – {max(durations):.2f}s")
            print(f"    Velocity range: {min(velocities)} – {max(velocities)}")
            print(f"    All strings:    {set(e['string'] for e in events)}")
            print(f"    Bow directions: {set(bow_dirs)}")
            print(f"    Total duration: {sum(durations):.1f}s "
                  f"({sum(durations)/60:.1f} min)")

            # Verify bow directions alternate
            dirs = [e['bow_dir'] for e in events]
            alternates = all(
                dirs[i] != dirs[i+1] for i in range(len(dirs)-1)
            )
            if alternates:
                print("    Bow directions alternate correctly (down/up/down/up...)")
            else:
                print("    Bow directions do not strictly alternate")
                print("      First 10:", dirs[:10])

    except Exception as e:
        print(f"    MIDI parse failed: {e}")
        all_pass = False

    # -- Check 5: Trajectory simulation ------------------------------
    print("\n[5] Simulating trajectory...")
    try:
        from baseline_controller import (
            parse_midi_a_string, velocity_to_force,
            velocity_to_speed, TIMESTEP_SEC
        )
        events = parse_midi_a_string(midi_path)

        bow_dir     = 'down'
        bow_pos     = 0.0   # start at frog
        max_bow_pos = 0.0
        min_bow_pos = 1.0
        total_steps = 0
        warnings    = []

        for i, note in enumerate(events):
            speed    = velocity_to_speed(note['velocity'])
            force    = velocity_to_force(note['velocity'])
            n_steps  = max(2, int(note['duration'] / TIMESTEP_SEC))
            distance = min(speed * note['duration'], BOW_LENGTH_A)
            fraction = distance / BOW_LENGTH_A

            if note['bow_dir'] == 'down':
                end_pos = min(bow_pos + fraction, 1.0)
            else:
                end_pos = max(bow_pos - fraction, 0.0)

            # Check for bow running out
            if note['bow_dir'] == 'down' and bow_pos + fraction > 0.95:
                warnings.append(
                    f"Note {i+1}: bow reaches {(bow_pos+fraction)*100:.0f}% "
                    f"(near tip) dur={note['duration']:.2f}s"
                )
            if note['bow_dir'] == 'up' and bow_pos - fraction < 0.05:
                warnings.append(
                    f"Note {i+1}: bow reaches {(bow_pos-fraction)*100:.0f}% "
                    f"(near frog) dur={note['duration']:.2f}s"
                )

            # Check force range
            if not (1.5 <= force <= 6.0):
                warnings.append(
                    f"Note {i+1}: force={force:.2f}N outside safe range [1.5, 6.0]"
                )

            max_bow_pos = max(max_bow_pos, end_pos)
            min_bow_pos = min(min_bow_pos, end_pos)
            bow_pos     = end_pos
            total_steps += n_steps

        print(f"    {total_steps} total timesteps ({total_steps*TIMESTEP_SEC:.1f}s)")
        print(f"    Bow position range: "
              f"{min_bow_pos*100:.0f}% – {max_bow_pos*100:.0f}%")

        if warnings:
            print(f"    {len(warnings)} warnings:")
            for w in warnings[:5]:
                print(f"      - {w}")
            if len(warnings) > 5:
                print(f"      ... and {len(warnings)-5} more")
        else:
            print("    All notes within safe bow range and force limits")

    except Exception as e:
        print(f"    Trajectory simulation failed: {e}")
        all_pass = False

    # -- Check 6: F/T sensor sanity ------------------------------
    print("\n[6] F/T sensor check...")
    ft = rtde_r.getActualTCPForce()
    ft_mag = float(np.linalg.norm(ft[:3]))
    print(f"    Current: fx={ft[0]:+.3f} fy={ft[1]:+.3f} fz={ft[2]:+.3f}")
    print(f"    Magnitude: {ft_mag:.3f}N")

    if ft_mag > 20.0:
        print("    Very large forces — sensor may not be tared or robot is in contact")
        print("      Tare sensor before running: rtde_c.zeroFtSensor()")
        all_pass = False
    elif ft_mag > 2.0:
        print("    Non-zero forces — bow may be in contact with string")
        print("      This is OK if bow is intentionally on string")
        print("      Tare with bow OFF string before data collection")
    else:
        print("    F/T sensor looks tared")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    if all_pass:
        print("ALL CHECKS PASSED — safe to run baseline")
        print("\nNext step:")
        print("  python3 baseline_controller.py <midi_file>")
    else:
        print("ISSUES FOUND — address warnings before running")
        print("  Do not run motion commands until resolved")
    print("="*60 + "\n")

    rtde_c.disconnect()
    rtde_r.disconnect()
    
    return all_pass


if __name__ == '__main__':
    import sys
    midi = sys.argv[1] if len(sys.argv) > 1 else \
           "../MIDI-Files/twinkle_twinkle-open.mid"
    preflight(midi)