import numpy as np

class QuaternionController:
    def __init__(self, J, Kp, Kd):
        self.J = J  # 3x3 matrix
        self.Kp = Kp
        self.Kd = Kd
        
    def quaternion_error(self, q_current, q_desired):
        q_c = np.array(q_current)
        q_d = np.array(q_desired)
        q_e = np.array([
            q_d[3]*q_c[0] - q_d[0]*q_c[3] - q_d[1]*q_c[2] + q_d[2]*q_c[1],
            q_d[3]*q_c[1] + q_d[0]*q_c[2] - q_d[1]*q_c[3] - q_d[2]*q_c[0],
            q_d[3]*q_c[2] - q_d[0]*q_c[1] + q_d[1]*q_c[0] - q_d[2]*q_c[3],
            q_d[3]*q_c[3] + q_d[0]*q_c[0] + q_d[1]*q_c[1] + q_d[2]*q_c[2]
        ])
        
        return q_e
    
    def compute_control_torque(self, q_current, q_desired, omega):
        q_current = np.array(q_current)
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
