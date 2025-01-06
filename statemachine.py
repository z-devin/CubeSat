# State Machine skeleton for cubesat
# Modes:
# - Detumbling Mode
# - Safe Mode
# - Free Drift Mode
# - Calibration Mode
# - Pointing Mode
# - Tracking Mode

import time
import threading
import numpy as np
from params import Context
from quaternion_controller import QuaternionController
from cube_visual import CubeVisualizer

class State:
    def __init__(self, name):
        self.name = name

    def handle_conditions(self, context):
        # for subclasses
        raise NotImplementedError("Subclasses must implement handle_conditions")
    
    def execute(self, context):
        # for subclasses
        raise NotImplementedError("Subclasses must implement execute")

class DetumblingState(State):
    def __init__(self):
        super().__init__("Detumbling Mode")

    def handle_conditions(self, context):
        if np.linalg.norm(context.omega_current) < 0.01:
            return SafeModeState()
        return self
    
    def execute(self, context):
        context.omega_current = np.array([0.005, 0.005, 0.005]) # need to implement detumble
        return self

class SafeModeState(State):
    def __init__(self):
        super().__init__("Safe Mode")

    def handle_conditions(self, context):
        if context.free_drift:
            return FreeDriftState()

        if np.linalg.norm(context.omega_current) > 0.01:
            return DetumblingState()
        
        if context.anomaly_detected:
            if context.calibration_mode:
                return CalibrationState()

        if not context.anomaly_detected:
            if context.tracking_target:
                return TrackingState()
            return PointingState()

        return self
    
    def execute(self, context):
        return self

class FreeDriftState(State):
    def __init__(self):
        super().__init__("Free Drift Mode")

    def handle_conditions(self, context):
        if not context.free_drift:
            return SafeModeState()
        return self
    
    def execute(self, context):
        return self

class CalibrationState(State):
    def __init__(self):
        super().__init__("Calibration Mode")

    def handle_conditions(self, context):
        if not context.calibration_mode:
            return SafeModeState()
        return self
    
    def execute(self, context):
        return self

class PointingState(State):
    def __init__(self):
        super().__init__("Pointing Mode")

        controller = QuaternionController(
            J=context.J,
            Kp=context.Kp,
            Kd=context.Kd
        )
        self.controller = controller

    def handle_conditions(self, context):
        if context.anomaly_detected:
            return SafeModeState()
        return self
    
    def execute(self, context):
        control_torque = self.controller.compute_control_torque(
            context.q_current, context.q_desired, context.omega_current
        )
        disturbance_torque = self.controller.compute_disturbance_torque(
            context.J, context.r, context.r_cp_s, context.A_s, context.C_r,
            context.n_hat, context.sun_vector, context.r_cp_a, context.rho,
            context.v, context.C_d, context.A_a, context.v_hat
        )
        total_torque = control_torque + disturbance_torque

        # updates
        omega_dot = np.linalg.inv(context.J) @ (total_torque - np.cross(context.omega_current, context.J @ context.omega_current))
        context.omega_current += omega_dot * 0.05  # omega update

        q_dot = 0.5 * np.array([
            context.q_current[3] * context.omega_current[0] - context.q_current[2] * context.omega_current[1] + context.q_current[1] * context.omega_current[2],
            context.q_current[2] * context.omega_current[0] + context.q_current[3] * context.omega_current[1] - context.q_current[0] * context.omega_current[2],
            -context.q_current[1] * context.omega_current[0] + context.q_current[0] * context.omega_current[1] + context.q_current[3] * context.omega_current[2],
            -context.q_current[0] * context.omega_current[0] - context.q_current[1] * context.omega_current[1] - context.q_current[2] * context.omega_current[2]
        ])
        context.q_current += q_dot * 0.05  # quaternion update
        context.q_current /= np.linalg.norm(context.q_current)  # nomralize


class TrackingState(State):
    def __init__(self):
        super().__init__("Tracking Mode")

    def handle_conditions(self, context):
        if context.anomaly_detected or not context.tracking_target:
            return SafeModeState()
        return self
    
    def execute(self, context):
        return self

class StateManager:
    def __init__(self):
        self.current_state = DetumblingState()

    def update_state(self, context):
        next_state = self.current_state.handle_conditions(context)
        if next_state != self.current_state:
            print(f"Transitioning from {self.current_state.name} to {next_state.name}")
            self.current_state = next_state

    def run(self, context):
        while True:
            self.update_state(context)
            self.current_state.execute(context)
            time.sleep(0.05)  #loop every 0.05s


if __name__ == "__main__":
    context = Context()
    visualizer = CubeVisualizer(context)
    manager = StateManager()

    state_machine_thread = threading.Thread(target=manager.run, args=(context,), daemon=True)
    state_machine_thread.start()

    visualizer.run()