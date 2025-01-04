# State Machine skeleton for cubesat
# Modes:
# - Detumbling Mode
# - Safe Mode
# - Free Drift Mode
# - Calibration Mode
# - Pointing Mode
# - Tracking Mode

class State:
    def __init__(self, name):
        self.name = name

    def handle_conditions(self, context):
        # for subclasses
        raise NotImplementedError("Subclasses must implement handle_conditions")
    
class DetumblingState(State):
    def __init__(self):
        super().__init__("Detumbling Mode")

    def handle_conditions(self, context):
        if context.angular_velocity < 0.01:
            return SafeModeState()
        return self

class SafeModeState(State):
    def __init__(self):
        super().__init__("Safe Mode")

    def handle_conditions(self, context):
        if context.free_drift:
            return FreeDriftState()

        if context.angular_velocity > 0.01:
            return DetumblingState()
        
        if context.anomaly_detected:
            if context.calibration_mode:
                return CalibrationState()

        if not context.anomaly_detected:
            if context.tracking_target:
                return TrackingState()
            return PointingState()

        return self

class FreeDriftState(State):
    def __init__(self):
        super().__init__("Free Drift Mode")

    def handle_conditions(self, context):
        if not context.free_drift:
            return SafeModeState()
        return self

class CalibrationState(State):
    def __init__(self):
        super().__init__("Calibration Mode")

    def handle_conditions(self, context):
        if not context.calibration_mode:
            return SafeModeState()
        return self

class PointingState(State):
    def __init__(self):
        super().__init__("Pointing Mode")

    def handle_conditions(self, context):
        if context.anomaly_detected:
            return SafeModeState()
        return self

class TrackingState(State):
    def __init__(self):
        super().__init__("Tracking Mode")

    def handle_conditions(self, context):
        if context.anomaly_detected or not context.tracking_target:
            return SafeModeState()
        return self
    
class Context:
    def __init__(self, angular_velocity, anomaly_detected, tracking_target, free_drift=False, calibration_mode=False):
        self.angular_velocity = angular_velocity
        self.anomaly_detected = anomaly_detected
        self.tracking_target = tracking_target
        self.free_drift = free_drift
        self.calibration_mode = calibration_mode


class StateManager:
    def __init__(self):
        self.current_state = DetumblingState()

    def update_state(self, context):
        """
        Update the current state based on the context.
        """
        next_state = self.current_state.handle_conditions(context)
        if next_state != self.current_state:
            print(f"Transitioning from {self.current_state.name} to {next_state.name}")
            self.current_state = next_state


if __name__ == "__main__":
    manager = StateManager()
    context = Context(angular_velocity=0.05, anomaly_detected=False, tracking_target=False)

    # simulating state transitions
    for step in range(5):
        print(f"Step {step}: Current State = {manager.current_state.name}")
        
        if step == 1:
            context.angular_velocity = 0.005
        elif step == 2:
            context.tracking_target = True
        elif step == 3:
            context.anomaly_detected = True

        manager.update_state(context)