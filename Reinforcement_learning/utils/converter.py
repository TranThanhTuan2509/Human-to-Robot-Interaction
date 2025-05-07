from config import *
"""
  Converter for TransporterNet, from a normal text's form into TransporterNet's form
"""

def converter(user_input):
  pick_target, place_target = 'ground', 'ground'
  if 'Push' in user_input or 'Move' in user_input:
    pick_text, place_text = user_input.split(' to ')
    for name in PLACE_TARGETS.keys():
      if name in place_text:
        place_target = name
        break

  elif 'Put' in user_input:
    pick_text, place_text = user_input.split(' onto ')
    for name in PLACE_TARGETS.keys():
      if name in place_text:
        place_target = name
        break
  else:
    pick_text = user_input

  for name in PICK_TARGETS.keys():
    if name in pick_text:
      pick_target = name
      break

  user_input = f'Pick the {pick_target} and place it on the {place_target}'

  return user_input, pick_target, place_target
