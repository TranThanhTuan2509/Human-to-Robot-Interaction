from random import randint, choice
from config import *


"""
  Randomly initialize the environment for DRL training
"""

def random_env():
    # Random env config
    k1 = k2 = 1
    # Get pick and place items
    pick_items = list(PICK_TARGETS.keys())
    pick_items = np.random.choice(pick_items, size=k1, replace=False)

    # Ensure place_items doesn't overlap with pick_items
    place_items = list(PLACE_TARGETS.keys())
    for pick_item in pick_items:
        if pick_item in place_items:  # Check before removing
            place_items.remove(pick_item)
    # If we don't have enough place items after removal, adjust k2
    if len(place_items) < k2:
        k2 = len(place_items)
    place_items = np.random.choice(place_items, size=k2, replace=False)

    # Create config for environment
    config = {'pick': pick_items, 'place': place_items}

    # Random instruction
    instruction = choice(instruction_form)

    return config, instruction