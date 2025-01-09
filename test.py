import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Mock Context for testing
class TestContext:
    def __init__(self):
        # Initial quaternion (arbitrary orientation)
        self.q_current = np.array([0.3, 0.2, 0.1, 0.927])
        self.q_current /= np.linalg.norm(self.q_current)  # Normalize
        # Initial angular velocity (rad/s)
        self.omega_current = np.array([0.1, 0.1, 0.1])
        # Earth's magnetic field in inertial frame (arbitrary, Tesla)
        self.B = np.array([25e-6, -20e-6, 15e-6])
        # Satellite's moment of inertia (kg·m²) - example for a small-ish satellite
        self.J = np.diag([0.006, 0.008, 0.010])

# Mock B-dot Controller
class MockBdotController:
    def __init__(self, k, dt):
        self.k = k
        self.dt = dt
        self.previous_B = None  # We'll set this on the first call

    def compute_magnetic_moment_and_bdot(self, current_B):
        if self.previous_B is None:
            self.previous_B = current_B
            B_dot = np.zeros_like(current_B)
        else:
            B_dot = (current_B - self.previous_B) / self.dt
            self.previous_B = current_B

        magnetic_moment = -self.k * B_dot  # m = -k * Bdot
        return magnetic_moment, B_dot

# DetumblingState Class for Testing
class DetumblingState:
    def __init__(self, controller):
        self.name = "Detumbling Mode"
        self.controller = controller
        
        # Histories
        self.angular_velocity_history = []
        self.omega_x_history = []
        self.omega_y_history = []
        self.omega_z_history = []
        
        # New: store B, Bdot, magnetic_moment, and torque histories
        self.Bx_history = []
        self.By_history = []
        self.Bz_history = []
        
        self.Bdot_x_history = []
        self.Bdot_y_history = []
        self.Bdot_z_history = []
        
        self.m_x_history = []
        self.m_y_history = []
        self.m_z_history = []
        
        self.tau_x_history = []
        self.tau_y_history = []
        self.tau_z_history = []
        
        self.dt = controller.dt  # so we don’t hardcode 0.01 everywhere

    def execute(self, context):
        # 1) Normalize quaternion
        context.q_current /= np.linalg.norm(context.q_current)

        # 2) Rotate B_inertial -> B_body
        rotation_matrix = Rotation.from_quat(context.q_current).as_matrix()
        current_B = rotation_matrix @ context.B  # body-frame magnetic field

        # 3) Compute magnetic moment & Bdot
        magnetic_moment, B_dot = self.controller.compute_magnetic_moment_and_bdot(current_B)

        # 4) Compute torque: tau = m x B
        total_torque = np.cross(magnetic_moment, current_B)
        
        # 5) Ensure torque always opposes rotation
        for i in range(3):
            if total_torque[i] * context.omega_current[i] > 0:
                total_torque[i] = -total_torque[i]

        # 5) Rigid-body dynamics: omega_dot = J^-1 [tau - omega x (J omega)]
        omega_dot = np.linalg.inv(context.J) @ (
            total_torque - np.cross(context.omega_current, context.J @ context.omega_current)
        )
        
        # 6) Update angular velocity (Euler integration)
        context.omega_current += omega_dot * self.dt
        
        # 7) Log angular velocity
        self.angular_velocity_history.append(np.linalg.norm(context.omega_current))
        self.omega_x_history.append(context.omega_current[0])
        self.omega_y_history.append(context.omega_current[1])
        self.omega_z_history.append(context.omega_current[2])
        
        # 8) Update quaternion (scalar-last [x, y, z, w] format)
        qx, qy, qz, qw = context.q_current
        wx, wy, wz = context.omega_current
        
        q_dot = 0.5 * np.array([
            qw*wx - qz*wy + qy*wz,
            qz*wx + qw*wy - qx*wz,
            -qy*wx + qx*wy + qw*wz,
            -qx*wx - qy*wy - qz*wz
        ])
        context.q_current += q_dot * self.dt
        context.q_current /= np.linalg.norm(context.q_current)
        
        # 9) Store B, Bdot, m, torque for plotting
        self.Bx_history.append(current_B[0])
        self.By_history.append(current_B[1])
        self.Bz_history.append(current_B[2])
        
        self.Bdot_x_history.append(B_dot[0])
        self.Bdot_y_history.append(B_dot[1])
        self.Bdot_z_history.append(B_dot[2])
        
        self.m_x_history.append(magnetic_moment[0])
        self.m_y_history.append(magnetic_moment[1])
        self.m_z_history.append(magnetic_moment[2])
        
        self.tau_x_history.append(total_torque[0])
        self.tau_y_history.append(total_torque[1])
        self.tau_z_history.append(total_torque[2])

    # ---- Plot methods ----
    def plot_angular_velocity_components(self):
        time = [i * self.dt for i in range(len(self.omega_x_history))]
        plt.figure(figsize=(10, 6))
        plt.plot(time, self.omega_x_history, label="$\omega_x$")
        plt.plot(time, self.omega_y_history, label="$\omega_y$")
        plt.plot(time, self.omega_z_history, label="$\omega_z$")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (rad/s)")
        plt.title("Angular Velocity Components Over Time")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_angular_velocity_magnitude(self):
        time = [i * self.dt for i in range(len(self.angular_velocity_history))]
        plt.figure(figsize=(10, 6))
        plt.plot(time, self.angular_velocity_history, label="$||\omega||$")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (rad/s)")
        plt.title("Angular Velocity Magnitude Over Time")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_B_components(self):
        time = [i * self.dt for i in range(len(self.Bx_history))]
        plt.figure(figsize=(10, 6))
        plt.plot(time, self.Bx_history, label="$B_x$")
        plt.plot(time, self.By_history, label="$B_y$")
        plt.plot(time, self.Bz_history, label="$B_z$")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnetic Field in Body Frame (T)")
        plt.title("Body-Frame Magnetic Field Components Over Time")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_Bdot_components(self):
        time = [i * self.dt for i in range(len(self.Bdot_x_history))]
        plt.figure(figsize=(10, 6))
        plt.plot(time, self.Bdot_x_history, label="$\\dot{B}_x$")
        plt.plot(time, self.Bdot_y_history, label="$\\dot{B}_y$")
        plt.plot(time, self.Bdot_z_history, label="$\\dot{B}_z$")
        plt.xlabel("Time (s)")
        plt.ylabel("B-dot (T/s)")
        plt.title("Time Derivative of Body-Frame B Field")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_moment_components(self):
        time = [i * self.dt for i in range(len(self.m_x_history))]
        plt.figure(figsize=(10, 6))
        plt.plot(time, self.m_x_history, label="$m_x$")
        plt.plot(time, self.m_y_history, label="$m_y$")
        plt.plot(time, self.m_z_history, label="$m_z$")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnetic Moment (A·m^2)")
        plt.title("Commanded Magnetic Moment Components Over Time")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_torque_components(self):
        time = [i * self.dt for i in range(len(self.tau_x_history))]
        plt.figure(figsize=(10, 6))
        plt.plot(time, self.tau_x_history, label="$\\tau_x$")
        plt.plot(time, self.tau_y_history, label="$\\tau_y$")
        plt.plot(time, self.tau_z_history, label="$\\tau_z$")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (N·m)")
        plt.title("Magnetic Torque Components Over Time")
        plt.grid()
        plt.legend()
        plt.show()

# Test Script
if __name__ == "__main__":
    # 1) Initialize mock context and controller
    context = TestContext()
    controller = MockBdotController(k=6000, dt=0.01)  # Example controller gain
    detumbling_state = DetumblingState(controller)

    # 2) Simulate for a certain number of steps
    steps = 200000  # e.g., 20 seconds at dt=0.01
    for _ in range(steps):
        detumbling_state.execute(context)

    # 3) Plot the results
    detumbling_state.plot_angular_velocity_components()
    detumbling_state.plot_angular_velocity_magnitude()

    detumbling_state.plot_B_components()
    detumbling_state.plot_Bdot_components()

    detumbling_state.plot_moment_components()
    detumbling_state.plot_torque_components()