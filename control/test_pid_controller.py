from simulation import FastPixhawkQuadSim
from pid_controller import KinematicVisualServoController
from trajectory import StochasticTargetTrajectory
import matplotlib.pyplot as plt

import numpy as np

dt = 0.02
duration = 20.0

trajectory = StochasticTargetTrajectory(
    duration=duration + 2.0,  
    profile="evasive",
    seed=42,
    initial_pos=[12.0, 0.0, -2.0],
    initial_heading_rad=0.0
)

all_target_pos = []
all_interceptor_pos = []

controller = KinematicVisualServoController()

simulator = FastPixhawkQuadSim(dt=dt)

simulator.reset(initial_pos=[0.0, 0.0, -2.0])

total_duration = round(duration / dt)

for step in range(total_duration):

    t = step * dt

    target_pos = trajectory.get_position(t)

    # projekter target til kamera -> giver telemetry / errors
    telemetry = simulator.get_camera_telemetry(target_pos)

    cmd = controller.compute_cmd(telemetry)

    simulator.step(cmd)

    all_interceptor_pos.append(simulator.pos.copy())

    all_target_pos.append(target_pos)

all_target_pos = np.array(all_target_pos)
all_interceptor_pos = np.array(all_interceptor_pos)
    
x_target = all_target_pos[:, 0]
y_target = all_target_pos[:, 1]

x_interceptor = all_interceptor_pos[:, 0]
y_interceptor = all_interceptor_pos[:, 1]




plt.plot(x_target, y_target, label="Target", color="red", alpha=0.5)
plt.plot(x_interceptor, y_interceptor, label="Interceptor", color="blue", alpha=0.8)
plt.legend()
plt.show()



