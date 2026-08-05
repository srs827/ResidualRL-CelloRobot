import rtde_control
import rtde_receive
import time

ROBOT_IP = "192.168.1.100"

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

TCP_OFFSET  = [0.028210348281514253, -0.09610723587300697,
               -0.09969041498611403, 0.0, 0.0, 0.0]
PAYLOAD_KG  = 0.260000
PAYLOAD_COG = [0.050000, -0.008000, 0.024000]

rtde_c.setTcp(TCP_OFFSET)
rtde_c.setPayload(PAYLOAD_KG, PAYLOAD_COG)

print("Before tare:")
print("  F/T:", [f"{v:+.3f}" for v in rtde_r.getActualTCPForce()])

# Tare the sensor
rtde_c.zeroFtSensor()
time.sleep(0.5)   # give it a moment to apply

print("After tare:")
print("  F/T:", [f"{v:+.3f}" for v in rtde_r.getActualTCPForce()])
print("  (should all be near 0.0 if bow is off string)")
print("  (should show contact forces if bow is on string)")

rtde_c.disconnect()
rtde_r.disconnect()