__author__ = "Gerald Würsching, Christian Pek"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "2024.1"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

import commonroad_rp.trajectories


class CostFunction(ABC):
    """
    Abstract base class for new cost functions
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
        """
        Computes the costs of a given trajectory sample
        :param trajectory: The trajectory sample for the cost computation
        :return: The cost of the given trajectory sample
        """
        pass


class DefaultCostFunction(CostFunction):
    """
    Default cost function for comfort driving
    """

    def __init__(self, desired_speed: Optional[float] = None, desired_d: float = 0.0,
                 desired_s: Optional[float] = None):
        super(DefaultCostFunction, self).__init__()
        # target states
        self.desired_speed = desired_speed
        self.desired_d = desired_d
        self.desired_s = desired_s

        # weights
        self.w_a = 5    # acceleration weight

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample, target_speed: Optional[float] = None):
        costs = 0.0
        # acceleration costs
        costs += np.sum((self.w_a * trajectory.cartesian.a) ** 2)
        # velocity costs
        if self.desired_speed is not None:
            costs += np.sum((5 * (trajectory.cartesian.v - self.desired_speed)) ** 2) + \
                     (50 * (trajectory.cartesian.v[-1] - self.desired_speed) ** 2) + \
                     (100 * (trajectory.cartesian.v[int(len(trajectory.cartesian.v)/2)] - self.desired_speed) ** 2)
        if self.desired_s is not None:
            costs += np.sum((0.25 * (self.desired_s - trajectory.curvilinear.s)) ** 2) + \
                 (20 * (self.desired_s - trajectory.curvilinear.s[-1])) ** 2

        # distance costs
        costs += np.sum((0.25 * (self.desired_d - trajectory.curvilinear.d)) ** 2) + \
                 (20 * (self.desired_d - trajectory.curvilinear.d[-1])) ** 2
        # orientation costs
        costs += np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2) + (
                5 * (np.abs(trajectory.curvilinear.theta[-1]))) ** 2

        return costs


class DefaultCostFunctionFailSafe(CostFunction):
    """
    Default cost function for fail-safe trajectory planning
    """

    def __init__(self):
        super(DefaultCostFunctionFailSafe, self).__init__()

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample):

        # acceleration costs
        costs = np.sum((1 * trajectory.cartesian.a) ** 2)
        # distance costs
        costs += np.sum((0.25 * trajectory.curvilinear.d) ** 2) + (20 * trajectory.curvilinear.d[-1]) ** 2
        # orientation costs
        costs += np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2) + (
                5 * (np.abs(trajectory.curvilinear.theta[-1]))) ** 2

        return costs

class WX1CostFunction(CostFunction):
    def __init__(self):
        super().__init__()
        # weights
        self.w_T = 10
        self.w_V = 1
        self.w_A = 0.1
        self.w_J = 0.0
        self.w_D = 0.1
        self.w_LC = 10

    def cost_time(self, traj: commonroad_rp.trajectories.TrajectorySample) -> float:
        return self.w_T - len(traj.cartesian.x) * traj.dt

    def cost_velocity_offset(self, vels: list, v_target: float) -> float:
        return self.w_V * np.sum(np.power(np.subtract(vels, v_target), 2))

    def cost_acceleration(self, accels: list) -> float:
        return self.w_A * np.sum(np.power(accels, 2))

    def cost_jerk(self, jerks: list) -> float:
        return self.w_J * np.sum(np.power(jerks, 2))

    def cost_lane_center_offset(self, offsets: list) -> float:
        return self.w_LC * np.sum(np.power(offsets, 2))

    def cost_total(self, traj: commonroad_rp.trajectories.TrajectorySample, target_speed: float) -> float:
        cost_time = self.cost_time(traj)
        cost_speed = self.cost_velocity_offset(traj.curvilinear.s_dot, target_speed)
        cost_accel = self.cost_acceleration(traj.curvilinear.s_ddot) + self.cost_acceleration(traj.curvilinear.d_ddot)
        # cost_jerk = self.cost_jerk(traj.curvilinear.s_ddd) + self.cost_jerk(traj.curvilinear.d_ddd)
        cost_offset = self.cost_lane_center_offset(traj.curvilinear.d)
        
        cost_total = (cost_time + cost_speed + cost_accel + cost_offset) / len(traj.cartesian.x)
        return cost_total

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample, target_speed: float = None) -> float:
        if target_speed is None:
            raise ValueError("Target speed must be provided for WX1CostFunction.")
        return self.cost_total(trajectory, target_speed)