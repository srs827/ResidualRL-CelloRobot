"""
test_servo_only.py
Tests servoL motion WITHOUT forceMode.
Verifies position control works before adding force control.
"""

import numpy as np
import time
import rtde_control
import rtde_receive
import atexit

ROBOT_IP = "192.168.1.100"

FROG_A = np.array([.336637615375,  .773335607743,  .103937252349,
                   -1.369835518384, -2.336199267621, 1.326965437172])
TIP_A  = np.array([.525205911288,  .350983193771,  .214779688012,
                   -1.369835518377, -2.336199267606, 1.326965437192])

TCP_OFFSET  = [0.028210348281514253, -0.09610723587300697,
               -0.09969041498611403, 0.0, 0.0, 0.0]
PAYLOAD_KG  = 0.260000
PAYLOAD_COG = [0.050000, -0.008000, 0.024000]

TIMESTEP_SEC = 0.05

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

atexit.register(rtde_c.disconnect)
atexit.register(rtde_r.disconnect)

rtde_c.setTcp(TCP_OFFSET)
rtde_c.setPayload(PAYLOAD_KG, PAYLOAD_COG)

print("Moving to frog...")
rtde_c.moveL(list(FROG_A), speed=0.05, acceleration=0.3)
time.sleep(0.5)

input("At frog. Press ENTER to run ONE slow stroke frog→tip (NO force mode)...")

# One stroke, no force mode at all
n_steps = 20
print(f"Running {n_steps} steps frog→tip...")

for i in range(n_steps):
    t   = i / (n_steps - 1)
    tcp = FROG_A + t * (TIP_A - FROG_A)

    rtde_c.servoL(
        tcp.tolist(),  # arg0: pose
        0.1,           # arg1: velocity
        0.1,           # arg2: acceleration
        TIMESTEP_SEC,  # arg3: time
        0.1,           # arg4: lookahead_time
        300,           # arg5: gain
    )

    state   = rtde_r.getActualTCPPose()
    bow_pos = float(np.dot(
        np.array(state[:3]) - FROG_A[:3],
        TIP_A[:3] - FROG_A[:3]
    ) / np.dot(TIP_A[:3] - FROG_A[:3], TIP_A[:3] - FROG_A[:3]))

    ft = rtde_r.getActualTCPForce()
    print(f"  step={i:2d}  bow={bow_pos:.3f}  "
          f"fx={ft[0]:+.2f}  fy={ft[1]:+.2f}  fz={ft[2]:+.2f}")

    time.sleep(TIMESTEP_SEC)

rtde_c.servoStop()
print("\nDone. Did bow position change from 0.00 to ~1.00?")
print("If yes: servoL works, problem is forceMode interaction")
print("If no:  servoL itself is not working — check robot state")