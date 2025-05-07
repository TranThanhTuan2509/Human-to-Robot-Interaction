def _init_touch_history(action_id, action_histories):
    action_histories[action_id] = {
        "ee_state1": [],
        "ee_state2": [],
        "ee_state3": [],
        "ee_state4": [],
        "height_history": [],
        "ee_state5": [],
        "ee_state6": [],
        "ee_state7": [],
        "joint_history": [],
        "contact_history": [],
        "task_state": {
            "action_type": None,
            "move_above_object": False,
            "move_down_to_object": False
        },
      "instruction": None,
      "pick_coordinate": None,
      "place_coordinate": None,
      "info": None
    }

    return action_histories[action_id]

def _init_pick_history(action_id, action_histories):
    action_histories[action_id] = {
        "ee_state1": [],
        "ee_state2": [],
        "ee_state3": [],
        "ee_state4": [],
        "height_history": [],
        "ee_state5": [],
        "ee_state6": [],
        "ee_state7": [],
        "joint_history": [],
        "contact_history": [],
        "task_state": {
            "action_type": None,
            "move_above_object": False,
            "move_down_to_object": False,
            "pick_up_object": False,
        },
      "instruction": None,
      "pick_coordinate": None,
      "place_coordinate": None,
      "info": None
    }

    return action_histories[action_id]

def _init_move_history(action_id, action_histories):
    action_histories[action_id] = {
        "ee_state1": [],
        "ee_state2": [],
        "ee_state3": [],
        "ee_state4": [],
        "height_history": [],
        "ee_state5": [],
        "ee_state6": [],
        "ee_state7": [],
        "joint_history": [],
        "contact_history": [],
        "task_state": {
            "action_type": None,
            "move_above_object": False,
            "move_down_to_object": False,
            "pick_up_object": False,
            "move_to_destination": False,
        },
      "instruction": None,
      "pick_coordinate": None,
      "place_coordinate": None,
      "info": None
    }

    return action_histories[action_id]

def _init_place_history(action_id, action_histories):
    action_histories[action_id] = {
        "ee_state1": [],
        "ee_state2": [],
        "ee_state3": [],
        "ee_state4": [],
        "height_history": [],
        "ee_state5": [],
        "ee_state6": [],
        "ee_state7": [],
        "joint_history": [],
        "contact_history": [],
        "task_state": {
            "action_type": None,
            "move_above_object": False,
            "move_down_to_object": False,
            "pick_up_object": False,
            "move_to_destination": False,
            "place_down_object": False,
        },
      "instruction": None,
      "pick_coordinate": None,
      "place_coordinate": None,
      "info": None
    }

    return action_histories[action_id]

def _init_push_history(action_id, action_histories):
    action_histories[action_id] = {
        "ee_state1": [],
        "ee_state2": [],
        "ee_state3": [],
        "ee_state4": [],
        "height_history": [],
        "ee_state5": [],
        "ee_state6": [],
        "ee_state7": [],
        "joint_history": [],
        "contact_history": [],
        "task_state": {
            "action_type": None,
            "move_above_object": False,
            "move_beside_to_object": False,
            "push_object": False
        },
      "instruction": None,
      "pick_coordinate": None,
      "place_coordinate": None,
      "info": None
    }

    return action_histories[action_id]

class ChooseBatch():
    def __init__(self):
        pass

    def _call_action_state(self, action_type, action_id, action_histories):
        """
        Updates sub-movements for an action based on env state.

        Args:
            action_type (str): 'touch', 'pick', 'move', 'place', 'push' as the actual action.
            action_id (int): Action ID in histories.
            action_histories (dict): Stores action histories.

        Returns:
            dict: updated action_histories.
        """
        # Initialize history
        init_funcs = {
            '0': _init_touch_history,
            '1': _init_pick_history,
            '2': _init_move_history,
            '3': _init_place_history,
            '4': _init_push_history
        }
        action_histories[action_id] = init_funcs[action_type](action_id, action_histories)
        action_histories[action_id]["task_state"]["action_type"] = action_type
        return action_histories[action_id]

    def assign_condition(self, state):
        if state["action_type"] == '0':
            if not state["move_above_object"]:
                state["move_above_object"] = True
            elif not state["move_down_to_object"]:
                state["move_down_to_object"] = True
            return state["move_down_to_object"], state

        elif state["action_type"] == '1':
            if not state["move_above_object"]:
                state["move_above_object"] = True
            elif not state["move_down_to_object"]:
                state["move_down_to_object"] = True
            elif not state["pick_up_object"]:
                state["pick_up_object"] = True
            return state["pick_up_object"], state

        elif state["action_type"] == '2':
            if not state["move_above_object"]:
                state["move_above_object"] = True
            elif not state["move_down_to_object"]:
                state["move_down_to_object"] = True
            elif not state["pick_up_object"]:
                state["pick_up_object"] = True
            elif not state["move_to_destination"]:
                state["move_to_destination"] = True
            return state["move_to_destination"], state

        elif state["action_type"] == '3':
            if not state["move_above_object"]:
                state["move_above_object"] = True
            elif not state["move_down_to_object"]:
                state["move_down_to_object"] = True
            elif not state["pick_up_object"]:
                state["pick_up_object"] = True
            elif not state["move_to_destination"]:
                state["move_to_destination"] = True
            elif not state["place_down_object"]:
                state["place_down_object"] = True
            return state["place_down_object"], state

        elif state["action_type"] == '4':
            if not state["move_above_object"]:
                state["move_above_object"] = True
            elif not state["move_beside_to_object"]:
                state["move_beside_to_object"] = True
            elif not state["push_object"]:
                state["push_object"] = True
            return state["push_object"], state
