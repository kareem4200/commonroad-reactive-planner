import os, sys
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import commonroad
from commonroad.visualization.mp_renderer import MPRenderer

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'model'))
sys.path.append(base_dir)
from model import CVAE, cvae_loss_function


class CVAEHelper:
      def __init__(self, scenario, planning_problem):
            # self.cvae_model = cvae_model
            # self.cvae_model.eval()
            
            # self.cvae_model = CVAE(X_dim=X_dim, c_dim=c_dim, z_dim=z_dim, h_Q_dim=h_Q_dim, h_P_dim=h_P_dim)
            # model.load_state_dict(torch.load('CVAE/model_weights/cvae_model.pth'))
            
            self.resnet_fe = resnet18 = models.resnet18(pretrained=True)
            self.resnet_fe = torch.nn.Sequential(*list(resnet18.children())[:-1])
            
            self._scenario = scenario
            self._planning_problem = planning_problem
            
            self._transform = transforms.Compose([
                  transforms.Resize((224, 224)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std =[0.229, 0.224, 0.225])
            ])
      # @property
      # def scenario(self):
      #       if self._scenario is None:
      #             raise ValueError("Scenario not set.")
      #       return self._scenario
      
      # @scenario.setter
      # def scenario(self, scenario):
      #       self._scenario = scenario
            
      # @property
      # def planning_problem(self):
      #       if self._pp is None:
      #             raise ValueError("Planning problem set not set.")
      #       return self._pp
            
      # @planning_problem.setter
      # def planning_problem(self, planning_problem):
      #       self._planning_problem = planning_problem

      def _build_cvae_condition(self, time_step):
            # Scenario has attribute position as a lanelet in goal (get the center of the lanelet) 
            if isinstance(self._planning_problem.goal.state_list[0].position, \
                  commonroad.geometry.shape.ShapeGroup):
                  goal_pos = self._planning_problem.goal.state_list[0].position.shapes[0].center
                  
            # Scenario has attribute position as a rectangle in goal (get the center of the rectangle)
            elif isinstance(self._planning_problem.goal.state_list[0].position, \
                  commonroad.geometry.shape.Rectangle):
                  goal_pos = self._planning_problem.goal.state_list[0].position.center
            
            condition = np.array([
                  self._planning_problem.initial_state.position[0],
                  self._planning_problem.initial_state.position[1],
                  self._planning_problem.initial_state.orientation,
                  self._planning_problem.initial_state.velocity,
                  goal_pos[0],
                  goal_pos[1]
            ])
            
            feature_vector = self.encode_scenario_image(time_step)
            condition.append(feature_vector)
            
            return condition

      def encode_scenario_image(self, time_step):
            """
            Renders a CommonRoad scenario and encodes the resulting image.
            """
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
            with torch.inference_mode():
                  features = self.resnet_fe(img_tensor).squeeze()  # shape: (512,)
            return features.numpy()