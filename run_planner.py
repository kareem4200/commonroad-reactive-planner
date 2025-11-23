__author__ = "Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "2024.1"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"


# standard imports
from copy import deepcopy
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
from commonroad_rp.utility.EvaluationCls import Evaluation
from commonroad_rp.trajectories import TrajectorySample, CartesianSample, CurviLinearSample

from commonroad_rp.utility.logger import initialize_logger
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=str, default="", help="scenario name to run on 1 scenario only")
parser.add_argument("--cvae", action=argparse.BooleanOptionalAction, help="--cvae or --no-cvae to enable/disable CVAE sampling")
parser.add_argument("--mode", type=str, default="train", help="mode: train, val, test")
args = parser.parse_args()

# *************************************
# Set Configurations
# *************************************

mode_dir = Path("../cvae/data/data_v2/") / args.mode
df = pd.read_parquet(mode_dir / f"c_{args.mode}.parquet")
scenarios_in_mode = df["scenario"].unique().tolist()

if args.cvae:
    config_file = "../cvae/config/reactive_planner_config_cvae.yaml"
else:
    config_file = "../cvae/config/reactive_planner_config_rp.yaml"
# scenarios_dir = "../cvae/scenarios/rp_success"
scenarios_dir = "../cvae/all_scenarios"

# initialize and get logger
logger = initialize_logger(ReactivePlannerConfiguration())

if args.scenario:
    scenarios = [args.scenario]
else:
    scenarios = os.listdir(scenarios_dir)

EVAL = False
# Initialize Planner
# *************************************

evaluation = Evaluation()

for i, sc in enumerate(scenarios):
    # --- initialize accumulators ---
    cart_x, cart_y, cart_theta, cart_v, cart_a = [], [], [], [], []
    cl_s, cl_s_dot, cl_s_ddot, cl_d, cl_d_dot, cl_d_ddot, cl_theta = [], [], [], [], [], [], []
    
    scenario_name = sc[:-4]
    
    if sc.endswith(".xml") and scenario_name in scenarios_in_mode:            
        
        time_list = []
        config = ReactivePlannerConfiguration.load(config_file, sc)
        config.update()
        
        evaluation.config = config
        
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
            
            # seed with initial state
            cart_x.append(planner.x_0.position[0])
            cart_y.append(planner.x_0.position[1])
            cart_theta.append(planner.x_0.orientation)
            cart_v.append(planner.x_0.velocity)
            cart_a.append(planner.x_0.acceleration)

            lon, lat, cl_theta0 = planner._compute_initial_states(planner.x_0)
            cl_s.append(lon[0])
            cl_s_dot.append(lon[1])
            cl_s_ddot.append(lon[2])
            cl_d.append(lat[0])
            cl_d_dot.append(lat[1])
            cl_d_ddot.append(lat[2])
            cl_theta.append(cl_theta0)

            while not planner.goal_reached():
                current_count = len(planner.record_state_list)

                planner.set_desired_velocity(current_speed=planner.x_0.velocity)

                time_start = time.time()
                # call plan function
                optimal, _ = planner.plan()

                time_list.append(time.time() - time_start)
                
                # append cartesian
                cart_x.append(optimal[4].x[1])
                cart_y.append(optimal[4].y[1])
                cart_theta.append(optimal[4].theta[1])
                cart_v.append(optimal[4].v[1])
                cart_a.append(optimal[4].a[1])

                # append curvilinear
                cl_s.append(optimal[5].s[1])
                cl_s_dot.append(optimal[5].s_dot[1])
                cl_s_ddot.append(optimal[5].s_ddot[1])
                cl_d.append(optimal[5].d[1])
                cl_d_dot.append(optimal[5].d_dot[1])
                cl_d_ddot.append(optimal[5].d_ddot[1])
                cl_theta.append(optimal[5].theta[1])
                
                # record planned state and input
                planner.record_state_and_input(optimal[0].state_list[1])
                
                # reset planner state for re-planning
                planner.reset(initial_state_cart=planner.record_state_list[-1], 
                            initial_state_curv=(optimal[2][1], optimal[3][1], 0.0), # arbitrary theta=0.0
                            collision_checker=planner.collision_checker, 
                            coordinate_system=planner.coordinate_system)
                
                if EVAL:
                    # Create ego vehicle and sampled trajectory bundle for visualization
                    if config.debug.show_plots or config.debug.save_plots:
                        ego_vehicle = planner.convert_state_list_to_commonroad_object(optimal[0].state_list)
                        sampled_trajectory_bundle = None
                        if config.debug.draw_traj_set:
                            sampled_trajectory_bundle = deepcopy(planner.stored_trajectories)
                    
                    if config.debug.show_plots or config.debug.save_plots:
                        visualize_planner_at_timestep(scenario=config.scenario, planning_problem=config.planning_problem,
                                                        ego=ego_vehicle, traj_set=sampled_trajectory_bundle,
                                                        ref_path=planner.reference_path, timestep=current_count, config=config)
                
            
            # save sampled variables and conditiobned variables if scenario is successfully planned
            if planner.goal_reached():
                # Create final cart_sample and cvln_sample
                cart_sample = CartesianSample(
                    x=np.array(cart_x),
                    y=np.array(cart_y),
                    theta=np.array(cart_theta),
                    v=np.array(cart_v),
                    a=np.array(cart_a),
                    kappa=np.zeros_like((len(cart_x),)),
                    kappa_dot=np.zeros_like((len(cart_x),)),
                    current_time_step=int(0)
                )
                cvln_sample = CurviLinearSample(
                    s=np.array(cl_s),
                    ss=np.array(cl_s_dot),
                    sss=np.array(cl_s_ddot),
                    d=np.array(cl_d),
                    dd=np.array(cl_d_dot),
                    ddd=np.array(cl_d_ddot),
                    theta=np.array(cl_theta),
                    current_time_step=int(0)
                )
                logger.info(f"Scenario {sc} successfully planned!")
                
                # run scenario evaluation
                if EVAL:
                    if config.sampling.cvae_sampling:
                        cvae_time_list = planner.sampling_space.cvae_inference_time_list
                    evaluation.build_trajectory_sample(cart_sample, cvln_sample)
                    evaluation.run_evaluation(planner.record_state_list,
                                              time_list,
                                              int(planner.num_sampled_trajectories),
                                              cvae_time_list if config.sampling.cvae_sampling else None,)
                    
                    make_gif(config, range(0, planner.record_state_list[-1].time_step))
                
        except Exception as e:
            logger.info(f"Scenario {sc} failed!")
            logger.info(f"Error: {e}")
            logger.info("Traceback:")
            logger.info(traceback.format_exc())
            continue

# save evaluation results to csv
if EVAL:
    evaluation.save_evaluation_results()
