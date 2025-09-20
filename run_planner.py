__author__ = "Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "2024.1"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"


# standard imports
from copy import deepcopy
import logging
import os
from termcolor import colored
import traceback

# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner

import sys
sys.path.insert(0, "/home/kareem/frenet_optimal_trajectory_planner/CVAE/commonroad-reactive-planner")

# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, make_gif
from commonroad_rp.utility.evaluation import run_evaluation
from commonroad_rp.utility.config import ReactivePlannerConfiguration

from commonroad_rp.utility.logger import initialize_logger
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=str, default="", help="scenario name to run on 1 scenario only")

args = parser.parse_args()
# *************************************
# Set Configurations
# *************************************
config_file = "../cvae/config/reactive_planner_config.yaml"
scenarios_dir = "../cvae/scenarios/rp_success"

# initialize and get logger
logger = initialize_logger(ReactivePlannerConfiguration())


# *************************************
# Initialize Planner
# *************************************
# run route planner and add reference path to config

time_list = []
if args.scenario:
    sc = args.scenario
    # print(f"Planning for {sc}")
    config = ReactivePlannerConfiguration.load(config_file, sc)
    config.update()
      
    try:
        # run route planner
        route_planner = RoutePlanner(config.scenario, config.planning_problem)
        route = route_planner.plan_routes().retrieve_first_route()
                
        # get reference path
        reference_path = route.reference_path
        
        planner = ReactivePlanner(config=config)

        # set reference path for curvilinear coordinate system
        planner.set_reference_path(route.reference_path)
        while not planner.goal_reached():
            current_count = len(planner.record_state_list)

            planner.set_desired_velocity(current_speed=planner.x_0.velocity)

            # call plan function
            optimal, _ = planner.plan()

            # record planned state and input
            planner.record_state_and_input(optimal[0].state_list[1])

            # reset planner state for re-planning
            planner.reset(initial_state_cart=planner.record_state_list[-1], 
                        initial_state_curv=(optimal[2][1], optimal[3][1]),
                        collision_checker=planner.collision_checker, 
                        coordinate_system=planner.coordinate_system)
            
            # Create ego vehicle and sampled trajectory bundle for visualization
            if config.debug.show_plots or config.debug.save_plots:
                ego_vehicle = planner.convert_state_list_to_commonroad_object(optimal[0].state_list)
                sampled_trajectory_bundle = None
                if config.debug.draw_traj_set:
                    sampled_trajectory_bundle = deepcopy(planner.stored_trajectories)
                        
            # visualize the current time step of the simulation
            if config.debug.show_plots or config.debug.save_plots:
                visualize_planner_at_timestep(scenario=config.scenario, planning_problem=config.planning_problem,
                                                ego=ego_vehicle, traj_set=sampled_trajectory_bundle,
                                                ref_path=planner.reference_path, timestep=current_count, config=config)
        
        # save sampled variables and conditiobned variables if scenario is successfully planned
        if planner.goal_reached():
            print(colored(f"Scenario {sc} successfully planned!", "green"))
            print(f"Planning took {time.time() - time_start} seconds")
            # avg_encode = sum(planner.sampling_space.cvae_helper._times) /\
            # len(planner.sampling_space.cvae_helper._times)
            # print(f"Average time to encode 1 scenario image is {avg_encode} seconds")
            # print(f"Number of samples: {planner.record_state_list[-1].time_step}")
            # save_scenario_imgs(sc[:-4], planner.record_state_list[-1].time_step)
            # make gif
            make_gif(config, range(0, planner.record_state_list[-1].time_step))

    except Exception as e:
        print(colored(f"Scenario {sc} failed!", "red"))
        print(f"Error: {e}")

else:
    for sc in os.listdir(scenarios_dir):
        if sc.endswith(".xml"):            
            
            config = ReactivePlannerConfiguration.load(config_file, sc)
            config.update()
            
            logger.info(f"Planning for {sc}")

            try:
                # run route planner
                route_planner = RoutePlanner(config.scenario, config.planning_problem)
                route = route_planner.plan_routes().retrieve_first_route()
                        
                # get reference path
                reference_path = route.reference_path
                
                planner = ReactivePlanner(config=config)

                # set reference path for curvilinear coordinate system
                planner.set_reference_path(route.reference_path)
                while not planner.goal_reached():
                    current_count = len(planner.record_state_list)

                    planner.set_desired_velocity(current_speed=planner.x_0.velocity)

                    # call plan function
                    optimal, _ = planner.plan()

                    # record planned state and input
                    planner.record_state_and_input(optimal[0].state_list[1])

                    # reset planner state for re-planning
                    planner.reset(initial_state_cart=planner.record_state_list[-1], 
                                initial_state_curv=(optimal[2][1], optimal[3][1]),
                                collision_checker=planner.collision_checker, 
                                coordinate_system=planner.coordinate_system)
                
                # save sampled variables and conditiobned variables if scenario is successfully planned
                if planner.goal_reached():
                    # print(colored(f"Scenario {sc} successfully planned!", "green"))
                    logger.info(f"Scenario {sc} successfully planned!")
                    # print(f"Number of samples: {planner.record_state_list[-1].time_step}")
                    # save_scenario_imgs(sc[:-4], planner.record_state_list[-1].time_step)
                    pass

            except Exception as e:
                # print(colored(f"Scenario {sc} failed!", "red"))
                logger.info(f"Scenario {sc} failed!")
                logger.info(f"Error: {e}")
                logger.info("Traceback:")
                logger.info(traceback.format_exc())
                continue

    # print(f"current time step: {current_count}")

# **************************
# Evaluate results
# **************************
evaluate = False
if evaluate:
    cr_solution, feasibility_list = run_evaluation(planner.config, planner.record_state_list, planner.record_input_list)
