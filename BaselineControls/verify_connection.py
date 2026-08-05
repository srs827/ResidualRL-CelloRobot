import rtde_control
import rtde_receive
import numpy as np

ROBOT_IP = "192.168.1.100"

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

print("TCP pose:  ", rtde_r.getActualTCPPose())
print("TCP force: ", rtde_r.getActualTCPForce())
print("Joints:    ", rtde_r.getActualQ())

# Set TCP offset and payload: make this match whichever script we are using
# ex. for A string 
TCP_OFFSET  = [0.028210348281514253, -0.09610723587300697,
               -0.09969041498611403, 0.0, 0.0, 0.0]
PAYLOAD_KG  = 0.260000
PAYLOAD_COG = [0.050000, -0.008000, 0.024000]

rtde_c.setTcp(TCP_OFFSET)
rtde_c.setPayload(PAYLOAD_KG, PAYLOAD_COG)

print("Setup complete.")