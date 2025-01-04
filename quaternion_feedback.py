# Quaternion Feedback Control Using PD Controller
# This code takes advantage of discretization to linearize the system locally

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *
from matplotlib.animation import FuncAnimation

class QuaternionController:
    def __init__(self, J, Kp, Kd):
        """
        Initialize the quaternion feedback controller.
        
        Args:
            J (ndarray): 3x3 inertia matrix
            Kp (float): Proportional gain
            Kd (float): Derivative gain
        """
        self.J = J
        self.Kp = Kp
        self.Kd = Kd
        
    def quaternion_error(self, q_current, q_desired):
        """
        Calculate quaternion error.
        
        Args:
            q_current (ndarray): Current quaternion [x,y,z,w]
            q_desired (ndarray): Desired quaternion [x,y,z,w]
            
        Returns:
            ndarray: Error quaternion
        """
        q_c = np.array(q_current)
        q_d = np.array(q_desired)
        
        # Quaternion multiplication for error calculation
        q_e = np.array([
            q_d[3]*q_c[0] - q_d[0]*q_c[3] - q_d[1]*q_c[2] + q_d[2]*q_c[1],
            q_d[3]*q_c[1] + q_d[0]*q_c[2] - q_d[1]*q_c[3] - q_d[2]*q_c[0],
            q_d[3]*q_c[2] - q_d[0]*q_c[1] + q_d[1]*q_c[0] - q_d[2]*q_c[3],
            q_d[3]*q_c[3] + q_d[0]*q_c[0] + q_d[1]*q_c[1] + q_d[2]*q_c[2]
        ])
        
        return q_e
    
    def compute_control_torque(self, q_current, q_desired, omega):
        """
        Compute control torque using quaternion feedback.
        """
        q_e = self.quaternion_error(q_current, q_desired)
        q_e_vec = q_e[:3]
        
        M_p = -self.Kp * q_e_vec
        M_d = -self.Kd * omega
        omega_cross_J = np.cross(omega, np.dot(self.J, omega))
        
        return M_p + M_d + omega_cross_J

    def compute_disturbance_torque(self, J, r, r_cp_s, A_s, C_r, n_hat, sun_vector, r_cp_a, rho, v, C_d, A_a, v_hat):
        """
        Compute disturbance torques: 
        - Gravity Gradient Torque
        Args:
            J (ndarray): 3x3 inertia matrix
            r (ndarray): 3x1 position vector from Earth's Center to Satellite

        - Solar Radiation Pressure Torque
        Args:
            r_cp_s (ndarray): 3x1 position vector from Earth's Center to Satellite
            A_s (float): Cross sectional area of the satellite
            C_r (float): Solar Radiation Pressure Coefficient
            n_hat (ndarray): 3x1 unit vector in the direction of the sun
            sun_vector (ndarray): 3x1 unit vector in the direction of the sun

        - Atmospheric Drag Torque
        Args:
            r_cp_a (ndarray): 3x1 position vector from Earth's Center to Satellite
            rho (float): Atmospheric Density
            v (float): Atmospheric Velocity
            C_d (float): Drag Coefficient
            A_a (float): Cross sectional area of the satellite
            v_hat (ndarray): 3x1 unit vector in the direction of the velocity    
            
        Returns:
            disturbance_torques (ndarray): 3x1 disturbance torques
        """
        # Gravity Gradient Torque
        # T_gravity = 3*mu/r^3 * (J*r_hat) x r_hat
        mu = 3.986e14                   # Gravitational Constant of Earth
        R = np.linalg.norm(r)           # Distance from Earth's Center to Satellite
        r_hat = r / np.linalg.norm(r)   # Unit vector from Earth's Center to Satellite
        T_gravity = (3 * mu / R**3) * np.cross(np.dot(J, r_hat), r_hat)

        # Solar Radiation Pressure Torque
        # T_solar = r_cp * F_solar
        # F_solar = P_solar * A * C_r * n_hat
        P_solar = 4.56e-6
        sun_vector = sun_vector / np.linalg.norm(sun_vector)
        projection = np.dot(n_hat, sun_vector)
        if projection <= 0:
            return np.zeros(3)
        F_solar = P_solar * A_s * C_r * n_hat
        T_solar = np.cross(r_cp_s, F_solar)

        # Atmospheric Drag Torque
        # T_drag = r_cp * F_drag
        # F_drag = 0.5 * rho * v^2 * C_d * A * v_hat
        v_hat = v_hat / np.linalg.norm(v_hat)
        F_drag_magnitude = 0.5 * rho * v**2 * C_d * A_a
        F_drag = -F_drag_magnitude * v_hat
        T_drag = np.cross(r_cp_a, F_drag)

        # Total Disturbance Torque
        disturbance_torques = T_gravity + T_solar + T_drag
        return disturbance_torques

    def system_dynamics(self, state, t, M, disturbance_torques):
        """
        System dynamics for simulation.
        """
        q = state[:4]
        omega = state[4:]
        
        q_dot = 0.5 * np.array([
            q[3]*omega[0] - q[2]*omega[1] + q[1]*omega[2],
            q[2]*omega[0] + q[3]*omega[1] - q[0]*omega[2],
            -q[1]*omega[0] + q[0]*omega[1] + q[3]*omega[2],
            -q[0]*omega[0] - q[1]*omega[1] - q[2]*omega[2]
        ])
        
        J_inv = np.linalg.inv(self.J)
        omega_dot = J_inv @ (M + disturbance_torques - np.cross(omega, self.J @ omega))
        
        return np.concatenate([q_dot, omega_dot])

def create_cube():
    """Create cube vertices and faces for plotting."""
    vertices = np.array([
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1]
    ])
    
    faces = np.array([
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # front
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # back
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # right
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # top
        [vertices[0], vertices[3], vertices[7], vertices[4]]   # bottom
    ])
    
    return faces

def plot_quaternion_error(states, q_desired, time):
    """
    Plot quaternion error over time in a 2D plot.

    Args:
        states (ndarray): Array of state vectors [q, omega] over time.
        q_desired (ndarray): Desired quaternion.
        time (ndarray): Time array corresponding to the states.
    """
    quaternion_errors = []
    
    for state in states:
        q_current = state[:4]
        # Compute quaternion error
        q_d = np.array(q_desired)
        q_c = np.array(q_current)
        
        # Quaternion multiplication for error calculation
        q_e = np.array([
            q_d[3]*q_c[0] - q_d[0]*q_c[3] - q_d[1]*q_c[2] + q_d[2]*q_c[1],
            q_d[3]*q_c[1] + q_d[0]*q_c[2] - q_d[1]*q_c[3] - q_d[2]*q_c[0],
            q_d[3]*q_c[2] - q_d[0]*q_c[1] + q_d[1]*q_c[0] - q_d[2]*q_c[3],
            q_d[3]*q_c[3] + q_d[0]*q_c[0] + q_d[1]*q_c[1] + q_d[2]*q_c[2]
        ])
        
        # Magnitude of the vector part of quaternion error
        q_e_magnitude = np.linalg.norm(q_e[:3])
        quaternion_errors.append(q_e_magnitude)

    plt.figure(figsize=(10, 6))
    plt.plot(time[:-1], quaternion_errors, label='Quaternion Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion Error Magnitude')
    plt.title('Quaternion Error vs. Time')
    plt.grid(True)
    plt.legend()
    plt.show()

def simulate_and_animate():
    """
    Simulate attitude control and create 3D animation.
    """
    # System parameters
    J = np.diag([100, 150, 80]) # Inertia matrix
    Kp = 100.0
    Kd = 100.0
    
    controller = QuaternionController(J, Kp, Kd)
    
    # Initial conditions (arbitrary for now)
    q_initial = np.array([0.3, 0.2, 0.1, 0.927])  # normalized arbitrary initial orientation
    q_initial = q_initial / np.linalg.norm(q_initial)
    omega_initial = np.array([0.1, -0.1, 0.1])
    state_initial = np.concatenate([q_initial, omega_initial])
    
    # Desired orientation (arbitrary for now)
    q_desired = np.array([0.0, 0.0, 0.0, 1.0])
    
    # Disturbance parameters (arbitrary for now)
    r = np.array([7000e3, 0, 0])  # Position vector from Earth's center to satellite (m)
    r_cp_s = np.array([0.1, 0, 0])  # Center of pressure for solar radiation (m)
    A_s = 0.01  # Cross-sectional area for solar radiation (m²)
    C_r = 1.5  # Reflectivity coefficient
    n_hat = np.array([1, 0, 0])  # Surface normal vector (aligned with x-axis)
    sun_vector = np.array([1.0, 0.5, 0.2])  # Arbitrary Sun direction vector

    r_cp_a = np.array([0.1, 0.1, 0])  # Center of pressure for atmospheric drag (m)
    rho = 1e-12  # Atmospheric density (kg/m³)
    v = 7500.0  # Velocity magnitude relative to atmosphere (m/s)
    C_d = 2.2  # Drag coefficient
    A_a = 0.01  # Cross-sectional area for atmospheric drag (m²)
    v_hat = np.array([1, 0, 0])  # Velocity direction unit vector

    # Simulation time
    t = np.linspace(0, 50, 2000)
    
    # Store states
    states = []
    current_state = state_initial
    
    for t_i in t[:-1]:
        # Compute control torque
        M = controller.compute_control_torque(
            current_state[:4],
            q_desired,
            current_state[4:]
        )

        # Compute disturbance torque
        M_disturbance = controller.compute_disturbance_torque(
            J, r, r_cp_s, A_s, C_r, n_hat, sun_vector, r_cp_a, rho, v, C_d, A_a, v_hat
        )
        M = M + M_disturbance
        
        # Integrate dynamics
        state_dot = controller.system_dynamics(current_state, t_i, M, M_disturbance)
        next_state = current_state + state_dot * (t[1] - t[0])
        next_state[:4] = next_state[:4] / np.linalg.norm(next_state[:4])
        
        states.append(current_state)
        current_state = next_state
    
    states = np.array(states)
    
    # Set up the animation
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    cube_faces = create_cube()
    
    def update(frame):
        ax.cla()
        # Set axis properties
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Get current quaternion
        q_current = states[frame, :4]
        # Convert quaternion to rotation matrix
        R = Rotation.from_quat(q_current).as_matrix()
        
        # Rotate and plot cube
        rotated_faces = np.array([[R @ vertex for vertex in face] for face in cube_faces])
        
        # Plot each face
        for face in rotated_faces:
            # Convert lists to 2D numpy arrays for plot_wireframe
            x = np.array([vertex[0] for vertex in face])
            y = np.array([vertex[1] for vertex in face])
            z = np.array([vertex[2] for vertex in face])
            
            # Close the face
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            z = np.append(z, z[0])
            
            # Reshape arrays to 2D
            x = x.reshape((1, -1))
            y = y.reshape((1, -1))
            z = z.reshape((1, -1))
            
            ax.plot_wireframe(x, y, z, color='blue')
        
        # Add local frame axes
        origin = np.array([0, 0, 0])  # Origin of local frame
        local_x = R @ np.array([1.5, 0, 0])  # Local x-axis
        local_y = R @ np.array([0, 1.5, 0])  # Local y-axis
        local_z = R @ np.array([0, 0, 1.5])  # Local z-axis

        # Draw axes
        ax.quiver(*origin, *local_x, color='red', label='Local X', arrow_length_ratio=0.1)
        ax.quiver(*origin, *local_y, color='green', label='Local Y', arrow_length_ratio=0.1)
        ax.quiver(*origin, *local_z, color='blue', label='Local Z', arrow_length_ratio=0.1)

        # Add title with current time
        ax.set_title(f'Time: {t[frame]:.2f} s')

    ani = FuncAnimation(fig, update, frames=len(states),
                    interval=50, repeat=True)
    plt.show()

    # Plot quaternion error over time
    plot_quaternion_error(states, q_desired, t)

    return states, t

if __name__ == "__main__":
    states, t = simulate_and_animate()