import numpy as np


class Scheduler:
    def __init__(self, s_per_degree, exp_t_per_proj: float = 0, n_proj_avg: float = 0):
        self.s_per_degree = s_per_degree
        self.exp_t_per_proj = exp_t_per_proj
        self.n_proj_avg = n_proj_avg
        self.current_t = 0

    def frame_avg(self) -> float:
        """Updates the current time to the next time.

        Returns:
            float: The next time.
        """
        self.current_t += self.exp_t_per_proj * self.n_proj_avg
        return self.current_t

    def rotate(self, current_angle: float, next_angle: float) -> float:
        """Updates the current time and angle to the next time and angle.

        Args:
            current_angle (float): The current angle.
            next_angle (float): The next angle.

        Returns:
            float: The next time.
        """
        self.current_t += self.s_per_degree * np.abs(next_angle - current_angle)
        return self.current_t

    def project(self) -> float:
        """Updates the current time to the next time.

        Returns:
            float: The next time.
        """
        self.current_t += self.exp_t_per_proj
        return self.current_t

    def reset_time(self) -> None:
        self.current_t = 0
