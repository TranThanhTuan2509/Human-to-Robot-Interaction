import numpy as np
from config import *
import pybullet
from utils.utils import *
import os

ROOT_PATH = os.path.abspath('.')

def root_path(target):
    return os.path.join(ROOT_PATH, 'asset', target)

class Reward:
    def __init__(self, correct_action, config=root_path('params/reward.xml'), max_steps=100,
                 robot_id=None, plane=None, ob1=None, ob2=None, suction_body=None, ee_link=None, dt=None, GOAL=None, env=None):
        """
        correct_action: the true action label (e.g., 'pick')
        threshold: threshold to consider sub-goal achieved
        max_steps: maximum steps per episode to prevent infinite loops
        robot_id: robot's joint id
        plane: ground id
        ob1: pick object id
        ob2: place object id
        """
        self.dt = dt
        self.correct = correct_action
        self.goal = GOAL
        self.step = 0  # for terminating
        self.max_steps = max_steps
        self.robot_id = robot_id
        self.plane = plane
        self.ob1 = ob1
        self.ob2 = ob2
        self.env = env
        self.ee_link = ee_link
        self.suction = suction_body
        self._read_specs_from_config(config)

    def _read_specs_from_config(self, robot_configs: str):
        """Read the specs of the reward params from the config xml file.
                - threshold0: success distance threshold.
                - threshold0: object distance threshold.
                - sigma0: scaling factor of the reward 0
                - sigma1: scaling factor of the reward 1
                - sigma2: scaling factor of the reward 2
                - sigma3: scaling factor of the reward 3

            Args:
                robot_configs (str): path to 'reward.xml'
            """
        root, root_name = get_config_root_node(config_file_name=robot_configs)
        robot_name = root_name[0]

        self.dist_threshold = read_config_from_node(
            root, "r" + self.correct, "threshold0", float
        )[0]
        self.obj_threshold = read_config_from_node(
            root, "r" + self.correct, "threshold1", float
        )[0]

        sigma = []
        for i in range(4):
            value = read_config_from_node(
                root, "r" + self.correct, f"sigma{i}", float
            )[0]
            sigma.append(value)

        self.sigma0, self.sigma1, self.sigma2, self.sigma3 = sigma

    def _check_collisions(self):
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
        self_collision = False
        ground_collision = False

        # --- Self-collision check (between different links only) ---
        # (contactFlags, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, ...)
        for contact in pybullet.getContactPoints(self.robot_id, self.robot_id):
            if contact[3] != contact[4]:  # Ignore same-link contacts
                collision_penalty += 0.1
                self_collision = True
                break  # early exit if collision already detected

        # --- suction-collision check (between different links only) ---
        for contact in pybullet.getContactPoints(self.suction.body, self.robot_id):
            if contact[3] != contact[4]:  # Ignore same-link contacts
                collision_penalty += 0.1
                self_collision = True
                break  # early exit if collision already detected

        # Ground collision check
        if pybullet.getContactPoints(self.robot_id, self.plane) or pybullet.getContactPoints(self.suction.body,
                                                                                             self.plane):
            ground_collision = True
            collision_penalty += 0.1

        # Object 1 collision checking (robot only except suction body)
        if self.ob1 is not None:
            if pybullet.getContactPoints(self.robot_id, self.ob1):
                pickable_object_collision = True
                collision_penalty += 0.1

        # Object 2 as target object collision checking
        if self.ob2 is not None:
            if pybullet.getContactPoints(self.robot_id, self.ob2):
                placable_object_collision = True
                collision_penalty += 0.1

            if pybullet.getContactPoints(self.suction.body, self.ob2):
                placable_object_collision = True
                collision_penalty += 0.1

        return collision_penalty, self_collision, ground_collision, pickable_object_collision, \
               placable_object_collision

    def _outside_workspace(self, coords):
        """
        Calculate workspace violation penalty based on distance outside bounds.
        Args:
            coords: List or tuple of [x, y] coordinates
            termination: object checking
        Returns:
            float: Penalty value (0 to -1.0)
        """

        x, y, z = coords
        bounds = BOUNDS.copy()
        ws_x, ws_y = 0.0, 0.0

        # X-axis violation
        if x > bounds[0][1]:
            ws_x = abs(x - bounds[0][1]) / 0.1
        elif x < bounds[0][0]:
            ws_x = abs(x - bounds[0][0]) / 0.1

        # Y-axis violation
        if y > bounds[1][1]:
            ws_y = abs(y - bounds[1][1]) / 0.1
        elif y < bounds[1][0]:
            ws_y = abs(y - bounds[1][0]) / 0.1

        total_violation = ws_x + ws_y
        return total_violation if total_violation > 0 else 0.0, True if total_violation > 0 else False

    def _check_angle_constraints(self, joints):
        """
        Check if any joint exceeds its angle limits.
        Args:
            joints: 6 joint angle values
        Returns:
            float: Penalty for exceeding joint limits (negative or 0)
            violation (bool): True if any joint limit is violated
        """
        total_penalty = 0.0
        violation = False

        for joint_idx, (lower_limit, upper_limit) in enumerate(JOINT_LIMITS):
            # joint = pybullet.getJointState(self.robot_id, joint_idx)
            current_angle = joints[joint_idx]  # Current joint angle (radians)
            # Check if angle is outside bounds
            if current_angle < lower_limit:
                total_penalty += (lower_limit - current_angle)
                violation = True
            elif current_angle > upper_limit:
                total_penalty += (current_angle - upper_limit)
                violation = True

        return abs(total_penalty), violation

    # velocity constrain
    def _check_suction_orientation(self):
        """Forcing to move suction with parallel - yaw pose"""
        ee_orientation = pybullet.getLinkState(self.robot_id, self.ee_link)[1]
        ee_euler = pybullet.getEulerFromQuaternion(ee_orientation)  # (roll, pitch, yaw)
        roll, pitch = ee_euler[0], ee_euler[1]
        roll_target = np.pi  # Vertical fingers
        pitch_target = 0.0
        roll_error = abs(roll - roll_target)
        pitch_error = abs(pitch - pitch_target)
        roll_penalty = roll_error if roll_error > np.deg2rad(5) else 0
        pitch_penalty = pitch_error if pitch_error > np.deg2rad(5) else 0
        return roll_penalty + pitch_penalty  # 10-degree tolerance

    def _align_velocity(self, current_obj_pos, previous_obj_pos, target):
        """Encourage moving towards goal
            Args:
                current_obj_pos: current object position
                previous_obj_pos: previous object position
                target: target destination
            Returns:
                float: reward/ penalty
        """
        # Velocity vector
        vel = (current_obj_pos - previous_obj_pos) / self.dt

        # Direction vectors (unit vectors)
        vel_norm = np.linalg.norm(vel)
        goal_dir = target - current_obj_pos
        goal_dir_norm = np.linalg.norm(goal_dir)

        if (vel_norm <= 0.015 and vel_norm >= 0.0) or (goal_dir_norm <= 0.015 and goal_dir_norm >= 0.0):
            return 0.0  # no movement or undefined direction

        vel_unit = vel / vel_norm
        goal_unit = goal_dir / goal_dir_norm

        alignment = np.dot(vel_unit, goal_unit)  # cos(theta) ∈ [-1, 1], expectation: 1

        return alignment

    def _check_inclination(self):
        """check object's pitch and roll inclination"""
        orientation = self.buf['obj_history'][-1][3:7]  # Adjust index based on your buffer structure
        euler = pybullet.getEulerFromQuaternion(orientation)
        pitch, roll = euler[1], euler[0]  # Scalars
        max_incl = np.deg2rad(10)  # 10 degrees
        incl = max(0, abs(pitch) - max_incl) + max(0, abs(roll) - max_incl)  # Corrected penalty

        return abs(incl)

    def _is_subgoal_achieved(self):
        """Check if the current sub-goal is achieved based on distance and contact state.
        Returns:
            bool: True if current task is successful else False
            """
        current_ee = self.buf['ee_history'][-1]  # Current end-effector state
        current_obj_up = self.buf['obj_history'][-1][:3]  # Current object position
        current_obj_down = self.buf['obj_history'][-1][7:]  # Current object position
        contact_condition = False
        contact = self.buf['contact_history']
        obj_po = np.linalg.norm(current_obj_down - self.goal)  # Tracking object position
        cur_dist = np.linalg.norm(current_ee - current_obj_up)

        # Sub-goals with contact conditions (e.g., suction activation)
        if self.correct in ['1', '2', '3']:  # For pick/move/place: suction should be active
            contact_condition = contact[-1][0] and contact[-1][1]
        elif self.correct in ['0', '4']:  # For touch/push: check success contact only
            contact_condition = contact[-1][1] and (not contact[-1][0])  # Check success contact only

        if self.correct in ['0']:
            return contact_condition
        elif self.correct in ['1', '2', '3', '4']:
            return obj_po <= self.obj_threshold and contact_condition and cur_dist < self.dist_threshold
        else:
            return False

    def get_reward(self, buffer, predicted_action: str, joints):
        """Compute reward and done flag for the current step.
        Args:
            buffer: a dict-like structure holding current and history states for the episode
            action_id: integer index for the current high-level action
            predicted_action: the agent's predicted action
            joints: 6 joint angle values
        Returns:
            float: reward/ penalty
            bool: done signal
        """
        self.buf = buffer
        self.predict = predicted_action
        self.step += 1
        obj_outs2 = 0.0
        reward_2 = 0.0
        reward_3 = 0.0
        outside2 = False

        # Penalize incorrect action prediction
        if self.correct != self.predict:
            return -1.0, False

        # Check if episode exceeds max steps
        if self.step == self.max_steps:
            return -0.5, True  # penalty if still remaining sub-action

        # Retrieve data
        contact = self.buf['contact_history']
        prev_obj_up = self.buf['obj_history'][-2][:3]  # previous object position (top coordination)
        current_obj_up = self.buf['obj_history'][-1][:3]  # Current object position (top coordination)
        prev_obj_down = self.buf['obj_history'][-2][7:]  # previous object position (bottom coordination)
        current_obj_down = self.buf['obj_history'][-1][7:]  # Current object position (bottom coordination)
        target_obj = self.buf['target_obj']  # Target object position
        current_ee = self.buf['ee_history'][-1]  # Current suction head position
        prev_act = self.buf['action'][-2]  # previous predicted action
        current_act = self.buf['action'][-1]  # current predicted action

        # Reward
        obj_po = np.linalg.norm(current_obj_down - self.goal)  # Tracking ee-object position
        cur_dist = np.linalg.norm(current_ee - current_obj_up)   # Tracking object-target position
        act_regularization = np.linalg.norm(current_act - prev_act)  # Deviation of joints
        suc = (contact[-1][0] and contact[-1][1] if self.correct in ['1', '2', '3'] else contact[-1][1])  # Suction and contact state
        toward_goal_vel = self._align_velocity(current_obj_up, prev_obj_up, self.goal)  # instant velocity

        self.env.writer.add_scalar(
            "Offset/ee_obj", cur_dist, (self.env.episode - 1)*self.max_steps + (self.step - 1)
        )
        self.env.writer.add_scalar(
            "Offset/obj_target", obj_po, (self.env.episode - 1)*self.max_steps + (self.step - 1)
        )

        # Constraint
        punished_suc = 0.3 if contact[-1][0] and self.correct in ['0', '4'] else 0.0  # discourage activating ee for touch/push action
        agl, violation = self._check_angle_constraints(joints)  # penalize joint collisions
        # outs, _ = self._outside_workspace(current_ee, termination=False)  # penalize moving outside workspace
        obj_outs1, outside1 = self._outside_workspace(current_obj_up)  # penalize tossing outside workspace
        if target_obj: obj_outs2, outside2 = self._outside_workspace(target_obj[-1][:3])  # penalize tossing outside workspace
        incl = self._check_inclination()  # object inclination
        col, collision1, collision2, collision3, collision4 = self._check_collisions()  # collision
        ee_pos = self._check_suction_orientation()  # penalize inclination suction
        self.env.writer.add_scalar(
            "Offset/Suction inclination", ee_pos, (self.env.episode - 1) * self.max_steps + (self.step - 1)
        )
        self.env.writer.add_scalar(
            "Offset/Object inclination", incl, (self.env.episode - 1) * self.max_steps + (self.step - 1)
        )

        # Compute total
        constraint = np.clip(
            col + agl + obj_outs1 + obj_outs2 + 0.4 * ee_pos + incl + punished_suc,
            0,
            1)

        reward_1 = (1.25 if self.env.episode < self.env.curriculum_learning else 1.25/4) * np.exp(-cur_dist / self.sigma0)  # max: 1.22 with offset of 0.2cm

        self.env.history['reward_1'].append(reward_1)
        # reward_4 = 0.3
        if suc and self.correct in ['1', '2', '3', '4']:
            # If suction is activated, robot maintains object on its suction.
            reward_2 = 2.5 * np.exp(-obj_po / self.sigma1)  # max: ~2.4 with offset of 1cm
            reward_3 = 0.2 * np.exp((toward_goal_vel / self.sigma2) - 1) if toward_goal_vel > 0 else 0.0  # max: 0.2

        # reward_4 = 0.25 * np.exp(-act_regularization / self.sigma3)
        total_reward = reward_1 + reward_2 + reward_3

        print(
            f'distance: {cur_dist} - '
            f'object: {obj_po} - '
            f'action: {act_regularization} - '
            f'suction: {suc} - '
            f'suction pose {ee_pos}'
              )
        print(
            f'\nrew distance: {reward_1} - '
            f'rew object: {reward_2} - '
            f'rew suction: {suc} - '
            f'rew towards {reward_3} - '
            f'constrain {constraint} - '
            f'collision {collision1 or collision2} - '
            f'outside {outside1 or outside2} - '
            f'joint violation {violation}'
        )

        if self._is_subgoal_achieved():
            total_reward = (reward_1 + 2.5 * 2.0 - constraint) if self.correct in ['1', '2', '3', '4'] else \
            (reward_1 * 2 - constraint)  # increase reward for completing all sub-goals
            return total_reward, True

        total_reward = total_reward - constraint

        return total_reward, outside1 or outside2 or violation or collision1 or collision2
