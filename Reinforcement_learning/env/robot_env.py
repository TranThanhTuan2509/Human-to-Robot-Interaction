import os
import numpy as np
import cv2
from utils.suction import Suction
import pybullet
import pybullet_data
from config import *
from reward.individual_reward import Reward
from gym.spaces import Discrete, Box
import gym
from utils.random_env import random_env
from models.TransporterNet import TransporterNetsEval
from models.text_encoder import Encoder
from random import choice
import time
from utils.converter import converter
import torch
from utils.choose_batch import ChooseBatch
'''
Index: 0, Name: world_joint
Index: 1, Name: rotated_base-base_fixed_joint

Index: 2, Name: shoulder_pan_joint
Index: 3, Name: shoulder_lift_joint
Index: 4, Name: elbow_joint
Index: 5, Name: wrist_1_joint
Index: 6, Name: wrist_2_joint
Index: 7, Name: wrist_3_joint

Index: 8, Name: ee_fixed_joint
Index: 9, Name: wrist_3_link-tool0_fixed_joint
Index: 10, Name: tool0_fixed_joint-tool_tip
Index: 11, Name: base_link-base_fixed_joint
'''
# To modify or change the objects in 3d platform we have to change the information in urdf file

''' FOR ABSOLUTE PATH '''

ROOT_PATH = os.path.abspath('..')

def root_path(target):
    return os.path.join(ROOT_PATH, '/asset', target)


def checkpoint(target):
  return os.path.join(ROOT_PATH, '/checkpoint', target)

# @markdown **Gym-style environment class:** this initializes a robot overlooking a workspace with objects.

class ImitationLearning(gym.Env):

  def __init__(self):
    self.dt = 1/480
    self.sim_step = 0
    # Configure and start PyBullet.
    # python3 -m pybullet_utils.runServer
    # pybullet.connect(pybullet.SHARED_MEMORY)  # pybullet.GUI for local GUI.
    pybullet.connect(pybullet.DIRECT)  # pybullet.GUI for local GUI.
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    pybullet.setPhysicsEngineParameter(enableFileCaching=0)
    assets_path = os.path.dirname(os.path.abspath(""))
    pybullet.setAdditionalSearchPath(assets_path)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.setTimeStep(self.dt)

    self.home_joints = (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0)  # Joint angles: (J0, J1, J2, J3, J4, J5).
    # self.home_joints = (0, 0, 0, 0, 0, 0)  # Joint angles: (J0, J1, J2, J3, J4, J5).
    self.home_ee_euler = (np.pi, 0, np.pi)  # (RX, RY, RZ) rotation in Euler angles.
    self.ee_link_id = 9  # Link ID of UR5 end effector.
    self.tip_link_id = 10  # Link ID of gripper finger tips.
    self.encoder = Encoder(cpu=True)
    self.transporterNet_checkpoint_path = checkpoint('165595')
    self.model = TransporterNetsEval(self.transporterNet_checkpoint_path)
    self.histories = ChooseBatch()
    self.suction = None
    # self.action_space = Discrete(6)
    self.action_space = Box(low=-3.0, high=3.0, shape=(6,), dtype=np.float32)
    self.p_action_space = Discrete(5)
    self.suction_space = Discrete(2)
    self.observation_space = Box(low=-np.inf, high=np.inf, shape=(528,), dtype=np.float32)
    self.current_action_id = -1
    self.suction_state = None
    self.state = None
    self.save_image = "./images"

  def _load(self, config):
    ''' Generate object in env'''
    self.config = config #Required objects
    self.obj_ids = []
    self.obj_name_to_id = {}  # Object id
    self.cache_video = []  # Frame
    self.object_properties = {}  # Object properties (width, length)

    pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    pybullet.setGravity(0, 0, -9.8)

    # Temporarily disable rendering to load URDFs faster.
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    # Add robot.
    # pybullet.loadURDF(root_path("plane.urdf"), [0, 0, -0.001])
    self.plane = pybullet.loadURDF(root_path("plane.urdf"), [0, 0, -0.001])
    self.robot_id = pybullet.loadURDF(root_path("ur5e/ur5e.urdf"), [0, 0, 0], flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    self.ghost_id = pybullet.loadURDF(root_path("ur5e/ur5e.urdf"), [0, 0, -10])  # For forward kinematics.
    self.joint_ids = [pybullet.getJointInfo(self.robot_id, i) for i in range(pybullet.getNumJoints(self.robot_id))]
    self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pybullet.JOINT_REVOLUTE]

    # Move robot to home configuration.
    for i in range(len(self.joint_ids)):
      pybullet.resetJointState(self.robot_id, self.joint_ids[i], self.home_joints[i])
      # print(self.joint_ids)

    # Add workspace.
    plane_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
    plane_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
    plane_id = pybullet.createMultiBody(0, plane_shape, plane_visual, basePosition=[0, -0.5, 0])
    pybullet.changeVisualShape(plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])

    # Load objects according to config.
    obj_names = list(self.config['pick']) + list(self.config['place'])
    obj_xyz = np.zeros((0, 3))
    for obj_name in obj_names:
      if any(obj in obj_name for obj in list(obj_list.keys())):

        # Get random position 17cm+ from other objects.
        while True:
          rand_x = np.random.uniform(BOUNDS[0, 0] + 0.1, BOUNDS[0, 1] - 0.1)
          rand_y = np.random.uniform(BOUNDS[1, 0] + 0.1, BOUNDS[1, 1] - 0.1)
          rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
          if len(obj_xyz) == 0:
            obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
            break
          else:
            nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
            if nn_dist > 0.17:
              obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
              break

        object_color = COLORS[obj_name.split(' ')[0]]
        object_type = obj_name.split(' ')[1]
        object_position = rand_xyz.squeeze()
        if object_type == 'block':
          object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
          object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
          object_id = pybullet.createMultiBody(0.01, object_shape, object_visual, basePosition=object_position)

        elif object_type == 'bowl':
          object_position[2] = 0
          object_id = pybullet.loadURDF(root_path("bowl/bowl.urdf"), object_position, useFixedBase=1)

        elif object_type in obj_list.keys():
          object_position[2] = 0
          object_id = pybullet.loadURDF(root_path(f"ycb_assets/0{obj_list[object_type]}_{object_type}.urdf"), object_position, useFixedBase=0)
        aabb_min, aabb_max = pybullet.getAABB(object_id)
        x_min, y_min, z_min = aabb_min
        x_max, y_max, z_max = aabb_max

        # Calculate dimensions
        width = x_max - x_min  # Along x-axis
        length = y_max - y_min  # Along y-axis

        self.object_properties[object_id] = [width, length]
        self.obj_ids.append(object_id)
        # print(self.object_properties)

        pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
        self.obj_name_to_id[obj_name] = object_id

    self.suction = Suction(self.robot_id, self.ee_link_id, self.obj_ids)  # Gripper
    self.suction.release()

    # Re-enable rendering.
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    for _ in range(300):
      pybullet.stepSimulation()

    self.read_state()
    print('Environment reset: done.')
    return self.get_observation()

  def reset(self):
    '''Create observation, instruction, coordinate'''
    self.action_histories = {} #action history as an external buffer

    config, instruction = random_env()
    obs = self._load(config)

    # 1. Joint positions
    joints = self.home_joints  # shape: [n_joints]

    object1 = config["pick"][0]
    object2 = config["place"][0]
    # 2. Instruction (flatten from [1, 512] to [512, ])
    instruction = instruction.format(object1, object2) \
      if instruction.count("{}") == 2 else instruction.format(object1)
    text, _, _, = converter(instruction)  # convert normal text form to transporterNet text's form

    self.info = [action_state[instruction.split(" ")[0]], object1, object2]
    coordinate = self.model.eval(obs, text)  # TransporterNet (input: instruction, obs | output:pick, place affordance)
    instruction = self.encoder.encode(instruction) #shape: [1, 512]
    self.instr = instruction.squeeze()  # shape: [512, ]

    # 3. Coordinates
    self.pick = coordinate['pick']  # ori shape: [3,]
    if object2 in fixed_destination.keys():
        self.place = np.array(fixed_destination[object2])
    else:
        self.place = coordinate['place']  # ori shape: [3,]

    coord = np.concatenate((self.pick, self.place))  # shape: [6,]

    ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0]) #shape: [3,]
    suction_signal = float(len(pybullet.getContactPoints(bodyA=self.suction.body, linkIndexA=0)) > 0) #shape: scalar

    # Increase the id 1 unit after resetting or completing an action
    self.current_action_id += 1
    return np.concatenate((joints, self.instr, coord, [suction_signal], ee_xyz)) #joint, text, pick + place location [528, ]

  def check_contact(self, obj_id):
    """ Check if the end effector contacts the object """
    contact_pts = pybullet.getContactPoints(obj_id, self.robot_id)
    # check if the contact is on the fingertip(s)
    self.suction_state = np.array(contact_pts > 0)

  def check_collisions(self, obj_id1=None, obj_id2=None):
      """
      Check for collisions involving the robot:
      - self-collision
      - ground collision
      - collision with objects (optional)
      Returns:
          collision_penalty (float): total penalty from collisions
      """
      collision_penalty = 0.0
      pickable_object_collision = False
      placable_object_collision = False

      # --- Self-collision check (between different links only) ---
      # (contactFlags, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, ...)
      for contact in pybullet.getContactPoints(self.robot_id, self.robot_id):
          if contact[3] != contact[4]:  # Ignore same-link contacts
              collision_penalty += 1.5
              break  # early exit if collision already detected

      # Ground collision check
      if pybullet.getContactPoints(self.robot_id, self.plane):
          collision_penalty += 1.5

      # Object 1 collision check
      if obj_id1 is not None:
          if pybullet.getContactPoints(self.robot_id, obj_id1):
              pickable_object_collision = True
              collision_penalty += 1.5

      # Object 2 collision check
      if obj_id2 is not None:
          if pybullet.getContactPoints(self.robot_id, obj_id2):
              placable_object_collision = True
              collision_penalty += 1.5

      return collision_penalty, pickable_object_collision, placable_object_collision

  def read_state(self):
    '''Update joint states information'''
    jointStates = pybullet.getJointStates(self.robot_id, self.joint_ids)
    jointPoses = [x[0] for x in jointStates]
    # jointVelocity = [x[1] for x in jointStates]
    # self.state = np.hstack((np.array(jointPoses), np.array(jointVelocity)))
    self.state = np.array(jointPoses)
    # self.state = np.append(self.state, self._target)

  def is_outside_workspace(self, ee_coords):
      """
      Calculate workspace violation penalty based on distance outside bounds.
      Args:
          ee_coords: List or tuple of [x, y] coordinates
      Returns:
          float: Penalty value (0 to -1.5)
      """
      x, y, z = ee_coords
      bounds = BOUNDS
      ws_x, ws_y = 0.0, 0.0

      # X-axis violation
      if x > bounds[0][1]:
          ws_x = abs(x - bounds[0][1]) / 0.05
      elif x < bounds[0][0]:
          ws_x = abs(x - bounds[0][0]) / 0.05

      # Y-axis violation
      if y > bounds[1][1]:
          ws_y = abs(y - bounds[1][1]) / 0.05
      elif y < bounds[1][0]:
          ws_y = abs(y - bounds[1][0]) / 0.05

      total_violation = ws_x + ws_y
      return 1.5 * min(1.0, total_violation) if total_violation > 0 else 0.0

  def servoj(self, joints):
    """Move to target joint positions with position control.
    torque = Kp.(p_target-p_current) - Kd.q'
    Typically, choosing Kp = 0.1.Kd"""
    pybullet.setJointMotorControlArray(
      bodyIndex=self.robot_id,
      jointIndices=self.joint_ids,
      controlMode=pybullet.POSITION_CONTROL,
      targetPositions=joints,
      positionGains=[0.06]*6)

  def step(self, action):
      """Perform actions"""
      given_action = self.info[0] #actual actions
      angles = action[0][0] #joint angles
      possible_action_signal = action[0][1] #predicted action (touch, pick, ...)
      suction_signal = float(action[0][2]) #suction state
      pickable_object_id = None
      placable_object_id = None
      # Update/Load action's history
      if self.current_action_id not in list(self.action_histories.keys()) or \
              action[1] not in list(self.action_histories.keys()):
          # Create batch following to predicted action or given instruction
          # If id is not in the history keys, then creating a new one with correct action batch
          self.action_histories[self.current_action_id] = self.histories._call_action_state(action_type=given_action,
                                                                                            action_id=self.current_action_id,
                                                                                            action_histories=self.action_histories)
          action_id = self.current_action_id
      else:
          # Reuse id if it exists in history keys
          action_id = action[1]

      history = self.action_histories[action_id]
      task_state = history["task_state"]

      # Save responding instruction and object coordinate
      if history["instruction"] is None:
          history["instruction"] = self.instr
          history["pick_coordinate"] = self.pick
          history["place_coordinate"] = self.place
          history["info"] = self.info

      # Initialize useful variables
      pickable_object = history["info"][1]
      placable_object = history["info"][2]
      pickable_object_id = self.obj_name_to_id[pickable_object]
      if placable_object not in fixed_destination.keys():
        placable_object_id = self.obj_name_to_id[placable_object]

      pick_xyz, place_xyz = history["pick_coordinate"], history["place_coordinate"]
      object_width = self.object_properties[self.obj_name_to_id[pickable_object]][0]  # Moving distance - prepare to push object
      object_length = self.object_properties[self.obj_name_to_id[pickable_object]][1]  # Moving distance - prepare to push object
      suction_body = self.suction.body
      gen_frame = 10  # number of frame that you want to generate when transiting to next action
      beside_xyz = pick_xyz.copy() #beside coordinate
      hover_xyz = pick_xyz.copy() + np.float32([0, 0, 0.2]) #above coordinate
      pick_xyz[2] -= 0.06499 #aoffset up: 3.5mm (6.149mm), aoffset down: 3.5mm (6.849mm)
      # pick_xyz[2] -= 0.06399  # aoffset up: 3mm (6.099mm), aoffset down: 3mm (6.699mm)

      # Manually calculate appropriate positions to move beside the object
      dx = place_xyz[0] - pick_xyz[0]
      dy = place_xyz[1] - pick_xyz[1]
      eps = 1e-6
      dist = np.sqrt(dx ** 2 + dy ** 2) + eps
      u_x = dx / dist
      u_y = dy / dist
      k = (object_width * abs(u_x) + object_length * abs(u_y)) * 1
      beside_xyz[0] -= k * u_x
      beside_xyz[1] -= k * u_y
      beside_xyz[2] = -0.04

      if not history["joint_history"]:
          arm_joint_ctrl = self.home_joints  # Start from home position
          # print('home joint:', self.home_joints)
      else:
          arm_joint_ctrl = history["joint_history"][-1]  # Use last known position
          # print('previous/latest joint angles:', history["joint_history"][-1])
      # print('predicted angles:', angles)
      arm_joint_ctrl = arm_joint_ctrl + angles  # Treat angles as deltas
      # print('arm_joint_ctrl:', arm_joint_ctrl)

      self.servoj(arm_joint_ctrl) #move robot
      for _ in range(500): #Apply physics and update robot
          pybullet.stepSimulation()
          # time.sleep(1. / 480.) #GUI mode

      self.read_state() #update robot joint position
      # print('after moving joint:', self.state)
      self.suction.release() if not suction_signal else self.suction.activate() #suction de/activate
      for _ in range(240): #Apply physics and update robot
          pybullet.stepSimulation()
          # time.sleep(1. / 480.) #GUI mode

      # Get current data
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
      # joint_angles = [pybullet.getJointState(self.robot_id, i)[0] for i in range(pybullet.getNumJoints(self.robot_id))][2:8]
      # jointStates = pybullet.getJointStates(self.robot_id, self.joint_ids)
      # state = np.array(jointPoses)
      joint_angles = self.state #current joint angles shape: [6, 1]
      instruction = history["instruction"] #encoded given instruction shape: [512, 1]
      coord_tensor = np.concatenate((history["pick_coordinate"], history["place_coordinate"]))  # shape: [6, 1]

      # Store current data
      history["ee_state1"].append(np.linalg.norm(hover_xyz[:2] - ee_xyz[:2])) #Move above object (XY)
      history["ee_state2"].append(np.linalg.norm(pick_xyz - ee_xyz)) #Reach object (XYZ)
      history["ee_state3"].append(np.linalg.norm(place_xyz[:2] - ee_xyz[:2])) #Move to next destination (XY)
      place_xyz[2] = 0.04 #placing height
      history["ee_state4"].append(np.linalg.norm(place_xyz - ee_xyz)) #Move down ee (XYZ)
      history["height_history"].append(np.linalg.norm(0.2 - ee_xyz[-1])) #Check ee height (Z)

      history["ee_state5"].append(np.linalg.norm(beside_xyz[:2] - ee_xyz[:2])) #Move above object but beside it (XY)
      history["ee_state6"].append(np.linalg.norm(beside_xyz - ee_xyz)) #Hold ee beside object (XYZ)
      place_xyz[2] = -0.04 #pushing height
      history["ee_state7"].append(np.linalg.norm(place_xyz - ee_xyz)) #Push object (XYZ)

      history["joint_history"].append(joint_angles) #joint angles
      history["contact_history"].append(float(len(pybullet.getContactPoints(bodyA=suction_body, linkIndexA=0)) > 0)) #Returns only contacts where linkIndexA == 0 in suction_body

      # Update sub-action states, guarantee action procedures
      last_sub_action, task_state = self.histories.assign_condition(task_state)
      history["task_state"] = task_state
      self.action_histories[action_id] = history
      # print('action_histories length:', len(self.action_histories.keys()))
      # Store frames to generate evaluation video
      # for _ in range(gen_frame):
      #     if self.current_action_id >= 50000:
      #         # self.cache_video.append(self.get_camera_image())
      #         image = cv2.imread(self.get_camera_image())
      #         # change the time in second
      #         cv2.imwrite(f'image_{time.time()}', image)

      # Update next state
      # observation = np.concatenate((joint_angles, instruction, coord_tensor)) #joint, text, pick + place location
      observation = np.concatenate((joint_angles, instruction, coord_tensor, [history["contact_history"][-1]], ee_xyz))
      # Compute reward
      reward_func = Reward(buffer=self.action_histories, instruction=history["info"][0],
                           state=possible_action_signal, action_id=action_id)
      reward = reward_func.get_reward()
      done = False

      #end effector moves outside workspace penalty
      reward -= self.is_outside_workspace(ee_xyz)

      # collision penalty and collision termination
      penalty, collision1, collision2 = self.check_collisions(pickable_object_id, placable_object_id)
      reward -= penalty
      if collision1 or collision2:
          done = True

      #last sub-action is performed
      if last_sub_action:
          self.current_action_id += 1
          done = True

      info = {}
      info['id'] = action_id # save id for next subsequent sub-actions
      return observation, reward, done, info

  def render(self):
    return self.get_observation()

  def get_observation(self):
    observation = {}

    # Render current image.
    color, depth, position, orientation, intrinsics = self.render_image()

    # Get heightmaps and colormaps.
    points = self.get_pointcloud(depth, intrinsics)
    position = np.float32(position).reshape(3, 1)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotation = np.float32(rotation).reshape(3, 3)
    transform = np.eye(4)
    transform[:3, :] = np.hstack((rotation, position))
    points = self.transform_pointcloud(points, transform)
    heightmap, colormap, xyzmap = self.get_heightmap(points, color, BOUNDS, PIXEL_SIZE)

    observation["image"] = colormap
    observation["xyzmap"] = xyzmap
    return observation

  def step_sim_and_render(self):
    pybullet.stepSimulation()
    self.sim_step += 1

    # Render current image at 8 FPS.
    if self.sim_step % (1 / (8 * self.dt)) == 0:
      self.cache_video.append(self.get_camera_image())

  def get_camera_image(self):
    # image_size = (1280, 1024)  # width, height
    # Adjust intrinsics for wider view:
    # fx = fy = 360  # lower than 720 for wider FOV
    # cx = image_size[0] / 2
    # cy = image_size[1] / 2
    image_size = (800, 800)
    fx = fy = 280  # lower than 800 for wider FOV
    cx = image_size[0] / 2
    cy = image_size[1] / 2

    intrinsics = (fx, 0, cx, 0, fy, cy, 0, 0, 1)
    color, _, _, _, _ = env.render_image(image_size, intrinsics)
    return color

  def set_alpha_transparency(self, alpha: float) -> None:
    for id in range(20):
      visual_shape_data = pybullet.getVisualShapeData(id)
      for i in range(len(visual_shape_data)):
        object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[i]
        rgba_color = list(rgba_color[0:3]) +  [alpha]
        pybullet.changeVisualShape(self.robot_id, linkIndex=i, rgbaColor=rgba_color)
        pybullet.changeVisualShape(self.suction.body, linkIndex=i, rgbaColor=rgba_color)

  def get_camera_image_top(self,
                           image_size=(240, 240),
                           intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
                           position=(0, -0.5, 5),
                           orientation=(0, np.pi, -np.pi / 2),
                           zrange=(0.01, 1.),
                           set_alpha=True):
    set_alpha and self.set_alpha_transparency(0)
    color, _, _, _, _ = env.render_image_top(image_size,
                                             intrinsics,
                                             position,
                                             orientation,
                                             zrange)
    set_alpha and self.set_alpha_transparency(1)
    return color

  def render_image_top(self,
                       image_size=(240, 240),
                       intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
                       position=(0, -0.5, 5),
                       orientation=(0, np.pi, -np.pi / 2),
                       zrange=(0.01, 1.)):

    # Camera parameters.
    orientation = pybullet.getQuaternionFromEuler(orientation)
    noise=False

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = position + lookdir
    focal_len = intrinsics[0]
    znear, zfar = (0.01, 10.)
    viewm = pybullet.computeViewMatrix(position, lookat, updir)
    fovh = (image_size[0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    aspect_ratio = image_size[1] / image_size[0]
    projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    # Render with OpenGL camera settings.
    _, _, color, depth, segm = pybullet.getCameraImage(
        width=image_size[1],
        height=image_size[0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        shadow=1,
        flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

    # Get color image.
    color_image_size = (image_size[0], image_size[1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # remove alpha channel
    if noise:
      color = np.int32(color)
      color += np.int32(np.random.normal(0, 3, color.shape))
      color = np.uint8(np.clip(color, 0, 255))

    # Get depth image.
    depth_image_size = (image_size[0], image_size[1])
    zbuffer = np.float32(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
    depth = (2 * znear * zfar) / depth
    if noise:
      depth += np.random.normal(0, 0.003, depth.shape)

    intrinsics = np.float32(intrinsics).reshape(3, 3)
    return color, depth, position, orientation, intrinsics

  def render_image(self, image_size=(720, 720), intrinsics=(360., 0, 360., 0, 360., 360., 0, 0, 1)):

    # Camera parameters.
    position = (0, -0.85, 0.4)
    orientation = (np.pi / 4 + np.pi / 48, np.pi, np.pi)
    orientation = pybullet.getQuaternionFromEuler(orientation)
    zrange = (0.01, 10.)
    noise=False

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = position + lookdir
    focal_len = intrinsics[0]
    znear, zfar = (0.01, 10.)
    viewm = pybullet.computeViewMatrix(position, lookat, updir)
    fovh = (image_size[0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    aspect_ratio = image_size[1] / image_size[0]
    projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    # Render with OpenGL camera settings.
    _, _, color, depth, segm = pybullet.getCameraImage(
        width=image_size[1],
        height=image_size[0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        shadow=1,
        flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    # depth value comes from camera
    # Get color image.
    color_image_size = (image_size[0], image_size[1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # remove alpha channel
    if noise:
      color = np.int32(color)
      color += np.int32(np.random.normal(0, 3, color.shape))
      color = np.uint8(np.clip(color, 0, 255))

    # Get depth image.
    depth_image_size = (image_size[0], image_size[1])
    zbuffer = np.float32(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
    depth = (2 * znear * zfar) / depth
    if noise:
      depth += np.random.normal(0, 0.003, depth.shape)

    intrinsics = np.float32(intrinsics).reshape(3, 3)
    return color, depth, position, orientation, intrinsics

  def get_pointcloud(self, depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points

  def transform_pointcloud(self, points, transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
      points: HxWx3 float array of 3D points in camera coordinates.
      transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
      points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
      points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points

  def get_heightmap(self, points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.
    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
      xyzmap: HxWx3 float array of XYZ points in world coordinates.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)
    xyzmap = np.zeros((height, width, 3), dtype=np.float32)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
      colormap[py, px, c] = colors[:, c]
      xyzmap[py, px, c] = points[:, c]
    colormap = colormap[::-1, :, :]  # Flip up-down.
    xv, yv = np.meshgrid(np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], height),
                         np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], width))
    xyzmap[:, :, 0] = xv
    xyzmap[:, :, 1] = yv
    xyzmap = xyzmap[::-1, :, :]  # Flip up-down.
    heightmap = heightmap[::-1, :]  # Flip up-down.
    return heightmap, colormap, xyzmap

env = ImitationLearning()
