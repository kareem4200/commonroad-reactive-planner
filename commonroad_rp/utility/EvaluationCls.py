from typing import List, Tuple, Dict

from commonroad.scenario.scenario import Scenario
from commonroad.common.solution import Solution, PlanningProblemSolution, VehicleModel, \
    VehicleType, CostFunction
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.trajectory import Trajectory

from commonroad_dc.feasibility.solution_checker import solution_feasible, starts_at_correct_state, obstacle_collision, \
    boundary_collision, ego_collision

from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.reactive_planner import ReactivePlannerState
from commonroad_rp.cost_function import DefaultCostFunction, WX1CostFunction
from commonroad_rp.trajectories import TrajectorySample, CartesianSample, CurviLinearSample
from commonroad_rp.polynomial_trajectory import QuinticTrajectory, QuarticTrajectory

import pandas as pd
import numpy as np
import os
import logging

class Evaluation:
    def __init__(self):
        self._config: ReactivePlannerConfiguration = None
        self._trajectory_sample: TrajectorySample = None
        self.cost_function = WX1CostFunction()
        self.records = []
        
        self.logger = logging.getLogger(__name__)
        
    @property
    def config(self) -> ReactivePlannerConfiguration:
        return self._config

    @config.setter
    def config(self, config: ReactivePlannerConfiguration):
        self._config = config
        
    @property
    def trajectory_sample(self) -> TrajectorySample:
        return self._trajectory_sample

    @trajectory_sample.setter
    def trajectory_sample(self, trajectory: TrajectorySample):
        self._trajectory_sample = trajectory
    
        
    def run_evaluation(self, 
                       state_list: List[ReactivePlannerState], 
                       planning_times: List[float],
                       num_samples: int,
                       cvae_inference_times: List[float] = None):
        
        ego_solution_trajectory = self.create_full_solution_trajectory(state_list)

        solution = self.create_planning_problem_solution(ego_solution_trajectory)
        valid, _ = self.valid_solution(solution)
        goal = self._config.planning_problem.goal
        if goal.state_list[0].has_value("velocity"):
            max_speed = goal.state_list[0].velocity.end
        else:
            max_speed = 13.5
        cost = self.cost_function.evaluate(trajectory=self.trajectory_sample, target_speed=max_speed)
        
        max_d_dev, avg_d_dev, std_d_dev = self.evaluate_d_deviation(trajectory=self.trajectory_sample)
        
        max_planning_time = np.max(planning_times)
        avg_planning_time = np.mean(planning_times)
        std_planning_time = np.std(planning_times)
        
        self.records.append({
            "scenario_name": str(self._config.scenario.scenario_id).strip(),
            "n_time_steps": int(len(planning_times)),
            # "valid_solution": valid,
            "total_cost": cost,
            "max_lateral_deviation": max_d_dev,
            "avg_lateral_deviation": avg_d_dev,
            "std_lateral_deviation": std_d_dev,
            "planning_time_max": max_planning_time,
            "planning_time_avg": avg_planning_time,
            "planning_time_std": std_planning_time,
            "num_sampled_trajectories": num_samples,
            "cvae_avg_inference_time": np.mean(cvae_inference_times) if cvae_inference_times is not None else None,
        })
    
    
    def save_evaluation_results(self, filename="evaluation_results.csv"):
        df = pd.DataFrame(self.records)
        path = os.path.join(self._config.general.path_output, filename)
        df.to_csv(path, index=False)
        self.logger.info(f"Saved evaluation results to {path}")
        
    def evaluate_d_deviation(self, trajectory: TrajectorySample):
        """
        Evaluate lateral deviation from desired d position
        """
        desired_d = 0.0
        d_deviation = []
        
        for d in trajectory.curvilinear.d:
            d_deviation.append(desired_d - d)

        return np.max(np.abs(d_deviation)), np.mean(np.abs(d_deviation)), np.std(np.abs(d_deviation))
        
        
    def create_full_solution_trajectory(self, state_list: List[ReactivePlannerState]) -> Trajectory:
        """
        Create CR solution trajectory from recorded state list of the reactive planner
        Positions are shifted from rear axis to vehicle center due to CR position convention
        """
        # convert initial state to ReactivePlannerState
        initial_state = self._config.planning_problem.initial_state
        
        if not hasattr(initial_state, 'acceleration'):
            initial_state.acceleration = 0.0
            
        initial_state_rp = initial_state.convert_state_to_state(ReactivePlannerState())
        initial_state_rp.steering_angle = np.arctan2(self._config.vehicle.wheelbase * initial_state_rp.yaw_rate, initial_state_rp.velocity)
    
        new_state_list = [initial_state_rp]
        
        for state in state_list:
            new_state_list.append(state.shift_positions_to_center(self._config.vehicle.wb_rear_axle))

        return Trajectory(initial_time_step=new_state_list[0].time_step, state_list=new_state_list)
        
        
    def create_planning_problem_solution(self, solution_trajectory: Trajectory) -> Solution:
        """
        Creates CommonRoad Solution object
        """
        pps = PlanningProblemSolution(planning_problem_id=self._config.planning_problem.planning_problem_id,
                                    vehicle_type=VehicleType(self._config.vehicle.id_type_vehicle),
                                    vehicle_model=VehicleModel.KS,
                                    cost_function=CostFunction.JB1,
                                    trajectory=solution_trajectory)

        # create solution object
        solution = Solution(self._config.scenario.scenario_id, [pps])
        return solution
    
    
    def build_trajectory_sample(self, 
                                cart_sample: CartesianSample,
                                cvln_sample: CurviLinearSample):
        """
        Build TrajectorySample from CartesianSample and CurviLinearSample and 2 arbitrary PolynomialTrajectory
        """
        lon_poly = QuarticTrajectory(x_0=np.ones((3, )), x_d=np.ones((3, )), delta_tau=1.0)
        lat_poly = QuinticTrajectory(x_0=np.ones((3, )), x_d=np.ones((3, )), delta_tau=1.0)
        
        trajectory = TrajectorySample(
            horizon=self._config.planning.time_steps_computation * self._config.planning.dt,
            dt=self._config.planning.dt,
            trajectory_long=lon_poly,
            trajectory_lat=lat_poly
        )
        
        trajectory.cartesian = cart_sample
        trajectory.curvilinear = cvln_sample
                
        self._trajectory_sample = trajectory
    
    
    def valid_solution(self, solution: Solution) -> Tuple[bool, Dict[int, Tuple[bool, Trajectory, Trajectory]]]:
        """
        Checks whether a solution is valid or not by checking
            - Solution has solved all planning problems of the scenario (solved_all_problems)
            - All planning problem solutions reached goal (goal_reached)
            - All planning problem solutions trajectories/input vectors start at correct time step (starts_at_correct_ts)
            - All planning problem solutions are feasible (solution_feasible)
            - There isn't a collision between the ego vehicles and the scenario obstacles (obstacle_collision)
            - The ego vehicles don't go out of lane boundaries (boundary_collision)
            - The ego vehicles don't collide with each other (ego_collision)

        It returns a dictionary that contains the inputs (or reconstructed inputs if trajectory solution), and
        trajectory (simulated trajectory if input vector solution).

        The valid_solution functions is being used as a basis for validity when a submission was made to
        `commonroad.in.tum.de website <https://commonroad.in.tum.de>`_ for evaluation. If the solution is not
        valid according to this function, then the solution will not be accepted.

        :param scenario: Scenario
        :param planning_problem_set: PlanningProblemSet
        :param solution: Solution
        :return: True if all checks pass successfully, and dictionary for simulated trajectories or reconstructed
            inputs. Raises SolutionCheckerException if any of the checks fail.
        """
        valid = all([
            starts_at_correct_state(solution, self.config.planning_problem_set),
            not obstacle_collision(self.config.scenario, self.config.planning_problem_set, solution),
            not boundary_collision(self.config.scenario, self.config.planning_problem_set, solution),
            not ego_collision(self.config.scenario, self.config.planning_problem_set, solution)
        ])
        results = solution_feasible(solution, self.config.scenario.dt, self.config.planning_problem_set)
        all_feasible = all([
            result[0]
            for pp_id, result in results.items()
        ])
        return valid and all_feasible, results
