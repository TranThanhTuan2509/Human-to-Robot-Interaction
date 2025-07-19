"""DRAFT: remove obj position and orientation"""
import os
import numpy as np
import random
from utils.suction import Suction
import pybullet
import pybullet_data
from reward.individual_reward import Reward
from gym.spaces import Discrete, Box
import gym
from utils.random_env import random_env
import torch
from config import *
from utils.utils import sample_point_on_top_of_arbitrary, world_to_local, local_to_world, read_config_from_node, get_config_root_node

ROOT_PATH = os.path.abspath('.')

def root_path(target):
    return os.path.join(ROOT_PATH, 'asset', target)

def checkpoint(target):
  return os.path.join(ROOT_PATH, 'checkpoint', target)

# @markdown **Gym-style environment class:** this initializes a robot overlooking a workspace with objects.

class ImitationLearning(gym.Env):

    def __init__(self, config=root_path('params/ur5.xml')):

        self.dt = 1/480 #smaller dt gets better accuracy
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
        self.home_ee_euler = (np.pi, 0, np.pi)  # (RX, RY, RZ) rotation in Euler angles.
        self.ee_link_id = 9  # Link ID of UR5 end effector.
        self.tip_link_id = 10  # Link ID of gripper finger tips.
        # self.encoder = Encoder(cpu=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.transporterNet_checkpoint_path = checkpoint('165595')
        # self.model = TransporterNetsEval(self.transporterNet_checkpoint_path)
        self.np_random = np.random.RandomState()
        self.suction = None
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype=np.float32) #rad/s
        self.p_action_space = Discrete(4)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32) #state space in range of (-3.1, 3.1)
        self.state = None
        self.action_type = None
        self.max_episode = None
        self.noise = True
        self.robot_noise_ratio = 0.01
        self.obj_noise = 0.005
        self.act_mid = np.zeros(7)
        self.act_rng = np.ones(7) * 2
        self.frame = 40
        self.true_episode = 0
        self.episode = 0
        self.writer = None
        self.sampled_local_point = None
        self.curriculum_learning = None
        self.max_steps = 100
        self.record = False
        self._read_specs_from_config(config)

    def seed(self, seed):
        print('set seed')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.np_random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True #GPU reproducibility
            torch.backends.cudnn.benchmark = False
        return [seed]

    def _load(self, config):
        ''' Generate object in env'''
        self.config = config #Required objects
        self.obj_ids = []
        self.obj_name_to_id = {}  # Object id
        self.object_properties = {}  # Object properties (width, length)
        # self.object_info = {} #Object info: x, y, z, ax, ay, az, vx, vy, vz

        pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
        pybullet.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load URDFs faster.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # Add robot.
        # pybullet.loadURDF(root_path("plane.urdf"), [0, 0, -0.001])
        self.plane = pybullet.loadURDF("plane.urdf", [0, 0, -0.001])
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
                    # Use TransporterNet for sim2real gap, provide precise position of object from reality
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
        for _ in range(200):
            pybullet.stepSimulation()
        # time.sleep(0.1)
        self._read_state()
        print('Environment reset: done.')

    def reset(self):
        '''Create observation, instruction, coordinate'''
        self._init_history()
        self.observation = {}
        self.cache_video = []  # for env visualization
        self.suction_value = []
        self.pickable_object_id = None
        self.placable_object_id = None

        config, instruction = random_env(self.action_type, seed=self.episode)
        self._load(config)

        # 1. Joint positions
        self._read_state()
        joints = self.state  # shape: [n_joints + velocities]
        self.joint = self.state[:6]

        object1 = config["pick"][0]
        object2 = config["place"][0]

        # 2. Instruction (flatten from [1, 512] to [512, ])
        instruction = instruction.format(object1, object2) \
            if instruction.count("{}") == 2 else instruction.format(object1)
        # text, _, _, = converter(instruction)  # convert normal text form to transporterNet text's form

        self.info = [action_state[instruction.split(" ")[0]], object1, object2]
        # Initialize reward and its information
        sampled_world = sample_point_on_top_of_arbitrary(self.obj_name_to_id[self.info[1]], margin=0.01, seed=self.episode)
        self.sampled_local_point = world_to_local(sampled_world, self.obj_name_to_id[self.info[1]])

        self.pickable_object_id = self.obj_name_to_id[self.info[1]]
        if object2 not in fixed_destination.keys():
            self.placable_object_id = self.obj_name_to_id[self.info[2]]
            self.ob2 = self.placable_object_id

        # masking factor for touching and picking
        self.instr = np.zeros(4, dtype=np.float32)
        self.instr[int(self.info[0])] = 1.0  # e.g., [0, 1, 0, 0] for action_id = 1
        # 3. Coordinates

        self.privilege = self._read_info(self.pickable_object_id)  # ori shape: [10,]

        if object2 in fixed_destination.keys():
            self.place = np.array(fixed_destination[object2])
            fix_destination = True
        else:
            self.place = self._read_info(self.placable_object_id)[:3]  # ori shape: [3,] + noise
            fix_destination = False

        self.place = self.place
        self.target = self._coor(self.privilege[:3], self.place, self.info[0], fix_destination)

        self.ee_ojb = np.float32(pybullet.getLinkState(self.suction.body, 0)[0]) - self.privilege[:3]  # relative distance between ee and obj
        self.obj_target = self.privilege[7:] - self.target  # relative distance between obj and target
        self.ee_target = np.float32(pybullet.getLinkState(self.suction.body, 0)[0]) - self.target

        self.coord = np.concatenate((self.ee_ojb, self.ee_target, self.privilege[3:7], self.obj_target))  # shape: [16,]
        self.coord += self._random_noise(self.coord, self.obj_noise)

        self.reward_func = Reward(correct_action=self.info[0], max_steps=self.max_steps, robot_id=self.robot_id,
                                  plane=self.plane, ob1=self.pickable_object_id, ob2=self.placable_object_id, suction_body=self.suction,
                                  ee_link=self.ee_link_id, dt=self.dt*self.frame, GOAL=self.target, env=self)

        self.observation = np.concatenate((self.instr,
                                         joints,  # joint position + joint velocity
                                         [0.0, 0.0, 0.0],  # suction converted value, suction state
                                         self.coord,
                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # previous action
                                         ))
        self.history["action"].append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.episode += 1
        if self.action_type in ['1', '2', '3']:
            self.true_episode += 1

        return self.observation

    def _do_simulation(self, frame: int):
        if self.record:
            for _ in range(frame):
                self.step_sim_and_render()
        else:
            for _ in range(frame):
                pybullet.stepSimulation()

    def step(self, action):
        """
        Perform actions
        Evaluate actions through its subsequent sub-actions
        """
        print('actual action: ', self.info[0])

        gen_frame = 2  # number of frame that you want to generate when transiting to next action
        self._read_state()
        # history["ee_history"].append(np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])) #shape: [3,]) store the old movement position
        self.history["ee_history"].append(np.float32(pybullet.getLinkState(self.suction.body, 0)[0]))  # shape: [3,])
        self.history["obj_history"].append(np.concatenate((
            local_to_world(self.sampled_local_point, self.obj_name_to_id[self.info[1]]),
            self._read_info(self.obj_name_to_id[self.info[1]])[3:])
                                                          ))
        self.history["action"].append(action)  # joint angles
        arm_joint_ctrl = np.clip(action, -1.0, 1.0)

        # Denormalize the input action from [-1, 1] range to the each actuators control range
        arm_joint_ctrl = self.act_mid + arm_joint_ctrl * self.act_rng

        # Compute single target position from converted velocities
        ctrl_feasible = self._ctrl_velocity_limits(arm_joint_ctrl, frame=self.frame)
        ctrl_feasible = self._ctrl_position_limits(ctrl_feasible)

        # Apply position control once
        self._servoj(ctrl_feasible[:6])
        # Simulate 40 steps
        self._do_simulation(self.frame)

        self._read_state()  # Final state after control

        suction_signal = float(ctrl_feasible[-1]) > 0.05  # suction state
        self.suction_value.append(float(ctrl_feasible[-1]))
        self.suction.activate() if suction_signal else self.suction.release()  # suction de/activate
        self._do_simulation(int(self.frame/2))

        print('suction state: ', suction_signal)
        # Get current data

        self.history["contact_history"].append([suction_signal, self._read_contact(self.info[0])]) #Returns only contacts where linkIndexA == 0 in suction_body
        self.history["ee_history"].append(np.float32(pybullet.getLinkState(self.suction.body, 0)[0])) #shape: [3,]) store the old ee position
        self.history["obj_history"].append(np.concatenate((
            local_to_world(self.sampled_local_point, self.obj_name_to_id[self.info[1]]),
            self._read_info(self.obj_name_to_id[self.info[1]])[3:])
        ))  # shape: [3,]) store new movement position
        if self.placable_object_id is not None:
            self.history["target_obj"].append(self._read_info(self.obj_name_to_id[self.info[2]])) # shape: [3,]) store new movement position

        self.ee_obj = np.float32(pybullet.getLinkState(self.suction.body, 0)[0]) - self._read_info(self.obj_name_to_id[self.info[1]])[:3]
        self.obj_target = self._read_info(self.obj_name_to_id[self.info[1]])[7:] - self.target
        self.ee_target = np.float32(pybullet.getLinkState(self.suction.body, 0)[0]) - self.target

        # Store frames to generate evaluation video
        if self.record:
            for _ in range(gen_frame):
                self.cache_video.append(self.get_camera_image())

        # Compute reward
        reward, done = self.reward_func.get_reward(buffer=self.history, predicted_action=str(self.info[0]),
                                                   joints=self.state[:6])

        self.coord = np.concatenate((self.ee_obj,
                                     self.ee_target,
                                     self._read_info(self.obj_name_to_id[self.info[1]])[3:7],
                                     self.obj_target
                                     ))
        self.coord += self._random_noise(self.coord, self.obj_noise)
        # Update next state
        self.observation = np.concatenate((self.instr,
                                           self.state,
                                           [float(ctrl_feasible[-1]),
                                            suction_signal,
                                            self._read_contact(self.info[0])],
                                           self.coord,
                                           self.history["action"][-1]
                                           ))  # 42D dimensional vector
        info = {}  # save id, privilege info for next subsequent sub-actions
        if self.action_type in ['3'] and done:
            self.suction.release()  # suction de/activate
            self._do_simulation(int(self.frame / 2))

        return self.observation, reward, done, info

    def _ctrl_velocity_limits(self, ctrl_velocity: np.ndarray, frame: int):
        """Enforce velocity limits and estimate joint position control input (to achieve the desired joint velocity).
        ALERT: This depends on previous observation. This is not ideal as it breaks MDP assumptions.
        Args:
            ctrl_velocity (np.ndarray): environment action with space: Box(low=-1.0, high=1.0, shape=(9,))

        Returns:
            ctrl_position (np.ndarray): input joint position given to the simulation actuators.
        """
        self._read_state() #update 'self.state' as the current robot joint angles
        ctrl_feasible_vel = np.clip(
            ctrl_velocity, self.robot_vel_bound[:7, 0], self.robot_vel_bound[:7, 1]
        )

        ctrl_feasible_position = np.pad(self.state[:6], (0, 1), mode='constant',
                                        constant_values=self.suction_value[-1] if self.suction_value else 0) + \
                                 ctrl_feasible_vel * self.dt * frame
        return ctrl_feasible_position

    #velocity-based target position computation
    def _ctrl_position_limits(self, ctrl_position: np.ndarray):
        """Enforce joint position limits.

        Args:
            ctrl_position (np.ndarray): unbounded joint position control input .

        Returns:
            ctrl_feasible_position (np.ndarray): clipped joint position control input.
        """
        ctrl_feasible_position = np.clip(
            ctrl_position, self.robot_pos_bound[:7, 0], self.robot_pos_bound[:7, 1]
        )
        return ctrl_feasible_position

    def _coor(self, pick_xyz, place_xyz, action, non_obj: bool):
        '''calculate touch, pick, move, place and push target'''
        hover_xyz = pick_xyz.copy() + np.float32([0, 0, 0.15])

        place_xyz_above = place_xyz.copy() + np.float32([0, 0, 0.15])

        place_xyz_down = place_xyz.copy()

        if action == '0':
            return pick_xyz
        elif action == '1':
            return hover_xyz
        elif action == '2':
            return place_xyz_above
        elif action == '3':
            if non_obj:
                return place_xyz
            else:
                return place_xyz_down
        else:
            return place_xyz

    def _random_noise(self, shape, noise):
        return np.concatenate([
            np.random.normal(-noise, noise, size=shape.shape[0])
        ])

    def _read_state(self):
        '''Update joint states information'''
        jointStates = pybullet.getJointStates(self.robot_id, self.joint_ids) #(position, velocity, reaction_forces, applied_joint_motor_torque)

        jointPoses = [x[0] for x in jointStates]
        jointVelocity = [x[1] for x in jointStates]
        jointTorque = [x[3] for x in jointStates]

        jointPoses += (
                self.robot_noise_ratio
                * self.robot_pos_noise_amp[:6]
                * self.np_random.uniform(low=-1.0, high=1.0, size=np.array(jointPoses).shape)
        )
        jointVelocity += (
                self.robot_noise_ratio
                * self.robot_vel_noise_amp[:6]
                * self.np_random.uniform(low=-1.0, high=1.0, size=np.array(jointVelocity).shape)
        )
        self.state = np.hstack((np.array(jointPoses).copy(), np.array(jointVelocity).copy()))

    def _read_info(self, obj_id):
        '''Update tracked object information'''
        pos, orn = pybullet.getBasePositionAndOrientation(obj_id)
        aabb_min, aabb_max = pybullet.getAABB(obj_id)
        linear_velocity, angular_velocity = pybullet.getBaseVelocity(obj_id)
        # dimensions = [max_v - min_v for min_v, max_v in zip(aabb_min, aabb_max)]  #object dimension: width, height, depth
        mass_data = pybullet.getDynamicsInfo(obj_id, -1)
        # mass = mass_data[0]
        center_coor_up = [(a + b) / 2 for a, b in zip(aabb_min, aabb_max)]
        center_coor_down = [(a + b) / 2 for a, b in zip(aabb_min, aabb_max)]
        pos = list(pos)

        center_coor_up[2] = aabb_max[
            2]  # the derivation of ee pose with object max_height for touch, pick, move and place action
        center_coor_down[2] = aabb_min[2]

        return np.concatenate((center_coor_up, orn, center_coor_down))  # shape: [10,]

    def _servoj(self, joints):
        """Move to target joint positions with position control.
        torque = Kp.(pt-pc) - Kd.q
        velocityGains = ?
        Directly move to desire angles"""
        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=joints,
            positionGains=[1] * 6)

    def _read_contact(self, action: str):
        """ Check if the end effector contacts correct object
        Args: action: action type
        Returns: bool
        """
        target_obj_id = self.obj_name_to_id[self.info[1]]

        if action in ['0', '1', '2', '3']:
            # Check suction head contact only
            points = pybullet.getContactPoints(bodyA=self.suction.body, linkIndexA=0)
            if points:
                for point in points:
                    obj_id = point[2]
                    if obj_id == target_obj_id:
                        return True
                return False
            return False
        else:
            # check entire suction body contact with target object (e.g., push)
            contacts_robot = pybullet.getContactPoints(bodyA=self.robot_id, bodyB=target_obj_id)
            contacts_base = pybullet.getContactPoints(bodyA=self.suction.base, bodyB=target_obj_id)
            contacts_head = pybullet.getContactPoints(bodyA=self.suction.body, bodyB=target_obj_id)
            return (
                    contacts_robot or
                    contacts_base or
                    contacts_head
            )

    def _read_specs_from_config(self, robot_configs: str):
        """Read the specs of the UR5 robot joints from the config xml file.
                - pos_bound: position limits of each joint.
                - vel_bound: velocity limits of each joint.
                - pos_noise_amp: scaling factor of the random noise applied in each observation of the robot joint positions.
                - vel_noise_amp: scaling factor of the random noise applied in each observation of the robot joint velocities.

            Args:
                robot_configs (str): path to 'ur5.xml'
            """
        root, root_name = get_config_root_node(config_file_name=robot_configs)
        robot_name = root_name[0]
        self.robot_pos_bound = np.zeros([self.action_space.shape[0], 2], dtype=float)
        self.robot_vel_bound = np.zeros([self.action_space.shape[0], 2], dtype=float)
        self.robot_pos_noise_amp = np.zeros(self.action_space.shape[0], dtype=float)
        self.robot_vel_noise_amp = np.zeros(self.action_space.shape[0], dtype=float)

        for i in range(self.action_space.shape[0]):
            self.robot_pos_bound[i] = read_config_from_node(
                root, "qpos" + str(i), "pos_bound", float
            )
            self.robot_vel_bound[i] = read_config_from_node(
                root, "qpos" + str(i), "vel_bound", float
            )
            self.robot_pos_noise_amp[i] = read_config_from_node(
                root, "qpos" + str(i), "pos_noise_amp", float
            )[0]
            self.robot_vel_noise_amp[i] = read_config_from_node(
                root, "qpos" + str(i), "vel_noise_amp", float
            )[0]

    def step_sim_and_render(self):
        pybullet.stepSimulation()
        self.sim_step += 1

        # Render current image every 10 time steps.
        if self.sim_step % (1 / (48 * self.dt)) == 0:
            self.cache_video.append(self.get_camera_image())

    def _init_history(self):
        self.history = {
            "obj_history": [],
            "ee_history": [],
            "action": [],
            "contact_history": [],
            "target_obj": [],
            "reward_1": []
        }

    def get_camera_image(self):
        image_size = (800, 800)
        fx = fy = 280  # lower than 800 for wider FOV
        cx = image_size[0] / 2
        cy = image_size[1] / 2

        intrinsics = (fx, 0, cx, 0, fy, cy, 0, 0, 1)
        color, _, _, _, _ = self.render_image(image_size, intrinsics)
        return color

    def render_image(self, image_size=(720, 720), intrinsics=(360., 0, 360., 0, 360., 360., 0, 0, 1), noise=False):
        # Camera parameters.
        position = (0, -0.85, 0.4)
        orientation = (np.pi / 4 + np.pi / 48, np.pi, np.pi)
        orientation = pybullet.getQuaternionFromEuler(orientation)
        zrange = (0.01, 10.)

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

