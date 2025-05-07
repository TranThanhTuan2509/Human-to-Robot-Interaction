"""
    Reward function, evaluate two things:
        - Correctly understand the instruction, matching the predicted action with the given instruction
        - Successfully complete an action
    Some definitions:
        "ee_state1": Move above object (XY),
        "ee_state2": Reach object (XYZ),
        "ee_state3": Move to next destination (XY),
        "ee_state4": Move down ee (XYZ),
        "height_history": Check ee height (Z),
        "ee_state5": Move above object but beside it (XY),
        "ee_state6": Hold ee beside object (XYZ),
        "ee_state7": Push object (XYZ),
        "joint_history": 6 joint-values,
        "contact_history": end effector state,
"""
class Reward:
    def __init__(self, buffer, instruction, state, action_id):
        """
        Args:
            buffer: a dictionary contains 2 smaller lists are:
                - ee: end effector coordinates
                - contact: contact state between ee and object surface
            instruction: given instruction for robot execution
            state: predicted action
        """
        self.buffer = buffer[action_id]
        # self.penalty_factor = 0.5
        self.thres1 = 0.011 #thres1hold
        self.thres2 = 0.0035 #thres2hold
        # self.thres2 = 0.003 #thres2hold
        self.instruction = instruction #correct action
        self.state = str(state[0]) #predicted action
        self.weights = [0.6, 0.4] #instruction, action

    def calculate_reward(self, sub_conditions, penalties):
        """
        Generic reward calculation based on sub-conditions and penalties.
        Args:
            sub_conditions (list of bool): List of conditions that must all be True for success.
            penalties (list of float): Penalty values for each failed condition.
        Returns:
            float: Reward value.
        """
        if all(sub_conditions):
            return 0.2 * len(sub_conditions) + 0.2  # Full success

        reward = 0.0
        for condition, penalty in zip(sub_conditions, penalties):
            if not condition:
                reward -= penalty
            else:
                reward += 0.2
        return reward

    def get_reward(self):
        """
        Compute reward only for the correct action if predicted matches instruction.
        Returns:
            tuple: (total_reward, done)
        """
        action_map = {
            '0': "touch", '1': "pick", '2': "move", '3': "place", '4': "push"
        }
        reward_funcs = {
            "touch": self.touch_reward,
            "pick": self.pick_reward,
            "move": self.move_reward,
            "place": self.place_reward,
            "push": self.push_reward
        }

        # Convert state to string for comparison
        predicted = action_map.get(self.state)
        correct = action_map.get(self.instruction)

        reward = reward_funcs[correct]()
        if predicted == correct and predicted in reward_funcs:
            # Compute only the reward for the matching action
            total_reward = self.weights[0] * 1.5 + self.weights[1] * reward
        else:
            # Penalty for incorrect action
            total_reward = self.weights[0] * -1.0 + self.weights[1] * reward

        return total_reward

    def touch_reward(self):
        # move above the object
        close_enough = self.buffer['ee_state1'][0] <= self.thres1 and self.buffer['height_history'][0] <= self.thres1 \
            if self.buffer["task_state"]["move_above_object"] else False
        penalty1 = (self.buffer['ee_state1'][0] + self.buffer['height_history'][0]) 

        # move down to reach object
        touch_closure = self.buffer['ee_state2'][1] <= self.thres2 \
            if self.buffer["task_state"]["move_down_to_object"] else False
        penalty2 = self.buffer['ee_state2'][1]  \
            if self.buffer["task_state"]["move_down_to_object"] else -0.03

        sub_conditions = [close_enough, touch_closure]
        penalties = [penalty1, penalty2]  # Penalties for each sub-condition
        # the highest fail value can reach
        return self.calculate_reward(sub_conditions, penalties)

    def pick_reward(self):
        # move above object
        close_enough = self.buffer['ee_state1'][0] <= self.thres1 and self.buffer['height_history'][0] <= self.thres1 \
            if self.buffer["task_state"]["move_above_object"] else False  # Distance between ee and the expected coordinate (move to expected destination)
        penalty1 = (self.buffer['ee_state1'][0] + self.buffer['height_history'][0]) 

        # move down to reach object
        touch_closure = self.buffer['ee_state2'][1] <= self.thres2 \
            if self.buffer["task_state"]["move_down_to_object"] else False # Distance between ee and the expected coordinate (reach)
        penalty2 = self.buffer['ee_state2'][1]  \
            if self.buffer["task_state"]["move_down_to_object"] else -0.03

        # activate suction
        activate = self.buffer['contact_history'][1] \
            if self.buffer["task_state"]["move_down_to_object"] else False#Concurrently activate the gripper at this state
        penalty3 = 0.5 if self.buffer["task_state"]["move_down_to_object"] else -0.03

        # pick up the object
        pick = self.buffer['ee_state1'][2] <= self.thres1 and self.buffer['height_history'][2] <= self.thres1  \
            if self.buffer["task_state"]["pick_up_object"] else False # Distance between ee and the expected coordinate (pick up)
        penalty4 = (self.buffer['ee_state1'][2] + self.buffer['height_history'][2])  \
            if self.buffer["task_state"]["pick_up_object"] else -0.03

        # remain activate suction to hold object
        remain_contact = self.buffer['contact_history'][2] \
            if self.buffer["task_state"]["pick_up_object"] else False#Concurrently activate the gripper at this state
        penalty5 = 0.5 if self.buffer["task_state"]["pick_up_object"] else -0.03

        sub_conditions = [close_enough, touch_closure, activate, pick, remain_contact]
        penalties = [penalty1, penalty2, penalty3, penalty4, penalty5]  # Penalties for each sub-condition
        # the highest fail value can reach
        return self.calculate_reward(sub_conditions, penalties)

    def move_reward(self):
        # move above the object
        close_enough = self.buffer['ee_state1'][0] <= self.thres1 and self.buffer['height_history'][0] <= self.thres1 \
            if self.buffer["task_state"]["move_above_object"] else False  # Distance between ee and the expected coordinate (move to expected destination)
        penalty1 = (self.buffer['ee_state1'][0] + self.buffer['height_history'][0]) 

        #move down to reach object
        touch_closure = self.buffer['ee_state2'][1] < self.thres2 \
            if self.buffer["task_state"]["move_down_to_object"] else False  # Distance between ee and the expected coordinate (reach)
        penalty2 = self.buffer['ee_state2'][1]  \
            if self.buffer["task_state"]["move_down_to_object"] else -0.03
        
        # activate suction
        activate = self.buffer['contact_history'][1] \
            if self.buffer["task_state"]["move_down_to_object"] else False  # Concurrently activate the gripper at this state
        penalty3 = 0.5 if self.buffer["task_state"]["move_down_to_object"] else -0.03

        # pick up the object
        pick = self.buffer['ee_state1'][2] <= self.thres1 and self.buffer['height_history'][2] <= self.thres1 \
            if self.buffer["task_state"]["pick_up_object"] else False  # Distance between ee and the expected coordinate (pick up)
        penalty4 = (self.buffer['ee_state1'][2] + self.buffer['height_history'][2])  \
            if self.buffer["task_state"]["pick_up_object"] else -0.03
        # remain activate suction to hold object
        remain_contact = self.buffer['contact_history'][2] \
            if self.buffer["task_state"]["pick_up_object"] else False  # Concurrently activate the gripper at this state
        penalty5 = 0.5 if self.buffer["task_state"]["pick_up_object"] else -0.03

        # move to new destination
        move_closure = self.buffer['ee_state3'][3] <= self.thres1 and self.buffer['height_history'][3] <= self.thres1 \
            if self.buffer["task_state"]["move_to_destination"] else False # Distance between ee and the expected coordinate (move)
        penalty6 = (self.buffer['ee_state3'][3] + self.buffer['height_history'][3])  \
            if self.buffer["task_state"]["move_to_destination"] else -0.03

        # remain holding object
        hold_object = self.buffer['contact_history'][3] \
            if self.buffer["task_state"]["move_to_destination"] else False#Concurrently activate the gripper at this state
        penalty7 = 0.5 if self.buffer["task_state"]["move_to_destination"] else -0.03

        sub_conditions = [close_enough, touch_closure, activate, pick, remain_contact, move_closure, hold_object]
        penalties = [penalty1, penalty2, penalty3, penalty4, penalty5, penalty6, penalty7]  # Penalties for each sub-condition
        # the highest fail value can reach
        return self.calculate_reward(sub_conditions, penalties)

    def place_reward(self):
        # move above object
        close_enough = self.buffer['ee_state1'][0] <= self.thres1 and self.buffer['height_history'][0] <= self.thres1 \
            if self.buffer["task_state"]["move_above_object"] else False  # Distance between ee and the expected coordinate (move to expected destination)
        penalty1 = (self.buffer['ee_state1'][0] + self.buffer['height_history'][0]) 

        # move down to reach object
        touch_closure = self.buffer['ee_state2'][1] <= self.thres2 \
            if self.buffer["task_state"]["move_down_to_object"] else False  # Distance between ee and the expected coordinate (reach)
        penalty2 = self.buffer['ee_state2'][1]  \
            if self.buffer["task_state"]["move_down_to_object"] else -0.03

        # activate suction
        activate = self.buffer['contact_history'][1] \
            if self.buffer["task_state"]["move_down_to_object"] else False  # Concurrently activate the gripper at this state
        penalty3 = 0.5 if self.buffer["task_state"]["move_down_to_object"] else -0.03

        # pick up the object
        pick = self.buffer['ee_state1'][2] <= self.thres1 and self.buffer['height_history'][2] <= self.thres1 \
            if self.buffer["task_state"]["pick_up_object"] else False  # Distance between ee and the expected coordinate (pick up)
        penalty4 = (self.buffer['ee_state1'][2] + self.buffer['height_history'][2])  \
            if self.buffer["task_state"]["pick_up_object"] else -0.03

        # remain the object
        remain_contact = self.buffer['contact_history'][2] \
            if self.buffer["task_state"]["pick_up_object"] else False  # Concurrently activate the gripper at this state
        penalty5 = 0.5 if self.buffer["task_state"]["pick_up_object"] else -0.03

        # move to new destination
        move_closure = self.buffer['ee_state3'][3] <= self.thres1 and self.buffer['height_history'][3] <= self.thres1 \
            if self.buffer["task_state"]["move_to_destination"] else False  # Distance between ee and the expected coordinate (move)
        penalty6 = (self.buffer['ee_state3'][3] + self.buffer['height_history'][3])  \
            if self.buffer["task_state"]["move_to_destination"] else -0.03

        # remain holding object
        hold_object = self.buffer['contact_history'][3] \
            if self.buffer["task_state"]["move_to_destination"] else False#Concurrently activate the gripper at this state
        penalty7 = 0.5 if self.buffer["task_state"]["move_to_destination"] else -0.03

        # move down to prepare release object
        place_closure = self.buffer['ee_state4'][4] <= self.thres1 \
            if self.buffer["task_state"]["place_down_object"] else False # Distance between ee and the expected coordinate (move)
        penalty8 = self.buffer['ee_state4'][4]  \
            if self.buffer["task_state"]["place_down_object"] else -0.03
        # release object
        release = not self.buffer['contact_history'][4] \
            if self.buffer["task_state"]["place_down_object"] else False#Concurrently activate the gripper at this state
        penalty9 = 0.5 if self.buffer["task_state"]["place_down_object"] else -0.03

        sub_conditions = [close_enough, touch_closure, activate, pick, remain_contact,
                          move_closure, hold_object, place_closure, release]
        penalties = [penalty1, penalty2, penalty3, penalty4, penalty5, penalty6, penalty7, penalty8, penalty9]  # Penalties for each sub-condition
        # the highest fail value can reach
        return self.calculate_reward(sub_conditions, penalties)

    def push_reward(self):
        # Move above object but beside it
        push_closure = self.buffer['ee_state5'][0] <= self.thres1 and self.buffer['height_history'][0] <= self.thres1 \
            if self.buffer["task_state"]["move_above_object"] else False # Distance between ee and the expected coordinate (reach)
        penalty1 = (self.buffer['ee_state5'][0] + self.buffer['height_history'][0]) if self.buffer["task_state"]["move_above_object"] else 0

        # move down beside object and check if suction can contact to the object then it's not moving to beside
        move_beside = self.buffer['ee_state6'][1] <= self.thres1 \
            if self.buffer["task_state"]["move_beside_to_object"] else False # Boolean indicating not contact with the object. (avoid collision between objects)
        penalty2 = self.buffer['ee_state6'][1]  \
            if self.buffer["task_state"]["move_beside_to_object"] else -0.03

        release = not self.buffer['contact_history'][1] if self.buffer["task_state"]["move_beside_to_object"] else False  # Boolean indicating not contact with the object. (avoid collision between objects)
        penalty3 = 0.5 if self.buffer["task_state"]["move_beside_to_object"] else -0.03

        # push object
        place_closure = self.buffer['ee_state7'][2] <= self.thres1 \
            if self.buffer["task_state"]["push_object"] else False # Distance between ee and the expected coordinate (move)
        penalty4 = self.buffer['ee_state7'][2]  \
            if self.buffer["task_state"]["push_object"] else -0.03

        sub_conditions = [push_closure, move_beside, release, place_closure]
        penalties = [penalty1, penalty2, penalty3, penalty4]  # Penalties for each sub-condition
        # the highest fail value can reach
        return self.calculate_reward(sub_conditions, penalties)
