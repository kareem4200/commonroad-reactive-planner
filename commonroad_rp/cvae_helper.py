import os, sys
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import commonroad
from commonroad.visualization.mp_renderer import MPRenderer

from cvae.model.model import CVAE, cvae_loss_function
from cvae.model.mask_background import MaskBackground

import time

class CVAEHelper:
    def __init__(self, scenario, planning_problem, mode):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # self.resnet_fe = torch.nn.Sequential(*list(resnet18.children())[:-1]).to(self.device)
        
        self._scenario = scenario
        self._planning_problem = planning_problem
        
        scenario_id = str(scenario.scenario_id).strip()
        
        self.conditions = pd.read_parquet("../cvae/data/data_v2/conditioned_vars.parquet")
        self.conditions = self.conditions.loc[self.conditions["scenario"] == scenario_id].copy()
        # self.encoded_imgs_df = pd.read_parquet("../cvae/data/data_extended/all_encoded_imgs.parquet")
        # self.scenario_imgs = self.encoded_imgs_df[self.encoded_imgs_df['scenario'] == str(scenario.scenario_id).strip()].copy()
        # self.scenario_imgs.drop(columns=['scenario'], inplace=True)
        # # self.encoded_imgs_df.reset_index(drop=True, inplace=True)
        # self.scenario_imgs.set_index('time_step', inplace=True)
        # self.scenario_imgs.sort_index(inplace=True)
        # print(self.scenario_imgs)
        
        if mode == "train":
            path = "../cvae/data/data_v2/train/imgs/"
        elif mode == "val":
            path = "../cvae/data/data_v2/val/imgs/"
        else:
            path = "../cvae/data/data_v2/test/imgs/"
        
        # x = pd.read_parquet(path + scenario_id)
        # self.no_time_steps = x.loc[x["scenario"] == str(scenario.scenario_id), "time_step"].max()
        # self.ego_id = x.loc[x["scenario"] == str(scenario.scenario_id) + ".xml", "ego_id"].iloc[0]
        
        self._transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                MaskBackground(),
        ])
        
        self.scenario_imgs = self.read_scenario_images(path + scenario_id)
        
        self._times = []


    def _build_cvae_condition(self, time_step, pre_encoded=True):
        start = time.time()
        # Scenario has attribute position as a lanelet in goal (get the center of the lanelet) 
        # if isinstance(self._planning_problem.goal.state_list[0].position, \
        #       commonroad.geometry.shape.ShapeGroup):
        #       goal_pos = self._planning_problem.goal.state_list[0].position.shapes[0].center
                
        # # Scenario has attribute position as a rectangle in goal (get the center of the rectangle)
        # elif isinstance(self._planning_problem.goal.state_list[0].position, \
        #       commonroad.geometry.shape.Rectangle):
        #       goal_pos = self._planning_problem.goal.state_list[0].position.center
        # print(f"Time step: {time_step}")
        # print(self.conditions.loc[self.conditions["time_step"] == time_step])
        condition = self.conditions.loc[self.conditions["time_step"] == time_step].iloc[0, 2:].to_numpy(dtype=np.float32)
        # print(condition.shape, condition)
        # SHIT: this is not updated to ego state each time step
        # condition = np.array([
        #         self._planning_problem.initial_state.position[0],
        #         self._planning_problem.initial_state.position[1],
        #         self._planning_problem.initial_state.orientation,
        #         self._planning_problem.initial_state.velocity,
        #         self._planning_problem.initial_state.acceleration,
        #         self._planning_problem.initial_state.yaw_rate,
        # ])
        
        if pre_encoded:
            feature_vector = self.read_feature_vector(time_step)
            condition = np.append(condition, feature_vector)
        else:
            # feature_vector = self.encode_scenario_image(time_step)
            # condition = np.append(condition, feature_vector)
            # self._times.append(time.time() - start)
            img = self.scenario_imgs[time_step]
        
        return condition, img

    def encode_scenario_image(self, time_step):
        """
        Renders a CommonRoad scenario and encodes the resulting image.
        """
        # start_time = time.time()
        # Draw the scenario with the renderer
        renderer = MPRenderer()
        renderer.draw_params.axis_visible = False
        renderer.draw_params.time_begin = time_step
        renderer.draw_params.dynamic_obstacle.draw_shape = True
        renderer.draw_params.dynamic_obstacle.draw_icon = True

        self._scenario.draw(renderer)
        self._planning_problem.draw(renderer)
        plt.gca().set_aspect("equal")
        renderer.render()

        # Capture figure as image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        buf.close()
        plt.close()

        # Transform and extract features
        img_tensor = self._transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        with torch.inference_mode():
                features = self.resnet_fe(img_tensor).squeeze()  # shape: (512,)
                
        # self._times.append(time.time() - start_time)
        print(features.to("cpu").numpy().shape, features.to("cpu").numpy())
        return features.to("cpu").numpy()
        
        
    def read_feature_vector(self, time_step):
        feature_vector = self.scenario_imgs.loc[time_step].values
        # print(f"Feature vector shape: {feature_vector.shape}, {feature_vector}")
        return feature_vector
    
    def read_scenario_images(self, path):
        imgs_path = os.listdir(path)
        imgs = []
        for img in imgs_path:
            with open(os.path.join(path, img), 'rb') as f:
                img = Image.open(f).convert('RGB')
                img_tensor = self._transform(img).unsqueeze(0)
                # img_tensor = img_tensor.to(self.device)
                imgs.append(img_tensor)
                
        return imgs