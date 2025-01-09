import numpy as np
import time

class Context:
    def __init__(self):
        # Earth's magnetic field
        self.B = np.array([25e-6, -20e-6, 15e-6])

        # Attitude and angular velocity
        self.q_current = np.array([0.3, 0.2, 0.1, 0.927])  # Initial quaternion
        self.q_current = self.q_current / np.linalg.norm(self.q_current)
        self.omega_current = np.array([0.1, 0.1, 0.1])     # Initial angular velocity
        
        # Desired orientation
        self.q_desired = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Inertia and control gains
        self.J = np.diag([0.008, 0.008, 0.010])            # Inertia matrix
        self.Kp = 0.006                                    # Proportional gain
        self.Kd = 0.007                                    # Derivative gain

        # Disturbance parameters
        self.r = np.array([7000e3, 0, 0])                  # Position vector from Earth's center
        self.r_cp_s = np.array([0.1, 0, 0])                # Center of pressure for solar radiation
        self.A_s = 0.01                                    # Solar radiation cross-sectional area
        self.C_r = 1.5                                     # Reflectivity coefficient
        self.n_hat = np.array([1, 0, 0])                   # Surface normal vector
        self.sun_vector = np.array([1.0, 0.5, 0.2])        # Sun direction vector

        self.r_cp_a = np.array([0.1, 0.1, 0])              # Center of pressure for drag
        self.rho = 1e-12                                   # Atmospheric density
        self.v = 7500.0                                    # Velocity relative to atmosphere
        self.C_d = 2.2                                     # Drag coefficient
        self.A_a = 0.01                                    # Drag cross-sectional area
        self.v_hat = np.array([1, 0, 0])                   # Velocity direction unit vector

        # Additional states
        self.anomaly_detected = False
        self.tracking_target = False
        self.free_drift = False
        self.calibration_mode = False