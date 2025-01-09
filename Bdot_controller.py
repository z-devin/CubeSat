import numpy as np

class BdotController:
    def __init__(self, k, dt):
        self.k = k
        self.dt = dt
        self.previous_B = None

    def compute_magnetic_moment(self, current_B):
        if self.previous_B is None:                 # avoids the spike since were using Bdot
            self.previous_B = current_B
            B_dot = np.zeros_like(current_B)
        else:
            B_dot = (current_B - self.previous_B) / self.dt
            self.previous_B = current_B

        magnetic_moment = -self.k * B_dot  # m = -k * Bdot
        return magnetic_moment