import rtde_receive
import atexit

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.100")
atexit.register(rtde_r.disconnect)

tcp = rtde_r.getActualTCPPose()
print("Full TCP pose:", tcp)
print("Position:    ", tcp[:3])
print("Orientation: ", tcp[3:])
print()
print("Use this as TASK_FRAME_A:")
print(f"TASK_FRAME_A = {list(tcp)}")