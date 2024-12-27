import numpy as np
from scipy.spatial.transform import Rotation
from scipy.integrate import odeint
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

    def system_dynamics(self, state, t, M):
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
        omega_dot = J_inv @ (M - np.cross(omega, self.J @ omega))
        
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

def simulate_and_animate():
    """
    Simulate attitude control and create 3D animation.
    """
    # System parameters
    J = np.diag([100, 150, 80]) # Inertia matrix
    Kp = 100.0
    Kd = 100.0
    
    controller = QuaternionController(J, Kp, Kd)
    
    # Initial conditions
    q_initial = np.array([0.3, 0.2, 0.1, 0.927])  # normalized arbitrary initial orientation
    q_initial = q_initial / np.linalg.norm(q_initial)
    omega_initial = np.array([0.1, -0.1, 0.1])
    state_initial = np.concatenate([q_initial, omega_initial])
    
    # Desired orientation (90-degree rotation about z-axis)
    q_desired = np.array([0.0, 0.0, 0.707, 0.707])
    
    # Simulation time
    t = np.linspace(0, 10, 200)
    
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
        
        # Integrate dynamics
        state_dot = controller.system_dynamics(current_state, t_i, M)
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
        
        # Add title with current time
        ax.set_title(f'Time: {t[frame]:.2f} s')

    ani = FuncAnimation(fig, update, frames=len(states),
                    interval=50, repeat=True)
    plt.show()
    return states, t

if __name__ == "__main__":
    states, t = simulate_and_animate()