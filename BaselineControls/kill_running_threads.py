"""
kill_rtde.py
Forcefully disconnects any hanging RTDE sessions.
Run this whenever a script didn't exit cleanly.
"""

import rtde_control
import rtde_receive
import time

ROBOT_IP = "192.168.1.100"

print("Connecting to kill hanging sessions...")

try:
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_c.stopScript()   # stops the ur-rtde control script on robot
    time.sleep(0.3)
    rtde_c.disconnect()
    print("Control interface killed.")
except Exception as e:
    print(f"Control interface: {e}")

try:
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_r.disconnect()
    print("Receive interface killed.")
except Exception as e:
    print(f"Receive interface: {e}")

print("Done. Robot should be free now.")