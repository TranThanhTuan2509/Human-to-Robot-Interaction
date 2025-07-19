import numpy as np

# Parameters for drawing figure.
display_input_size = (10, 10)
overall_fig_size = (18, 24)

# Parameters for drawing figure.
line_thickness = 1
fig_size_w = 35
# fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
mask_color = 'red'
alpha = 0.5

#@markdown **Global constants:** pick and place objects, colors, workspace bounds.
PIXEL_SIZE = 0.00267857 # The size of each pixel in real-world units (such as meters)

BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # (X, Y, Z) (width, depth, height)

# [-6.28318530718, -6.28318530718, -3.14159265359, -6.28318530718, -6.28318530718, -6.28318530718]
# [6.28318530718, 6.28318530718, 3.14159265359, 6.28318530718, 6.28318530718, 6.28318530718]
JOINT_LIMITS = [[- 2 * np.pi, 2 * np.pi], [- 2 * np.pi, 2 * np.pi], [-np.pi, np.pi],
						   [- 2 * np.pi, 2 * np.pi], [- 2 * np.pi, 2 * np.pi], [- 2 * np.pi, 2 * np.pi]]

obj_list = {'block': None, 'bowl': None, 'banana': '11', 'apple': '13', 'orange': '17'
	,'peach': '15', 'pear': '16', 'gelatin_box': '09', 'strawberry': '12'}

COLORS = {
	'blue': (78 / 255, 121 / 255, 167 / 255, 255 / 255),
	'red': (255 / 255, 87 / 255, 89 / 255, 255 / 255),
	'green': (89 / 255, 169 / 255, 79 / 255, 255 / 255),
	'orange': (242 / 255, 142 / 255, 43 / 255, 255 / 255),
	'yellow': (237 / 255, 201 / 255, 72 / 255, 255 / 255),
	'purple': (176 / 255, 122 / 255, 161 / 255, 255 / 255),
	'pink': (255 / 255, 157 / 255, 167 / 255, 255 / 255),
	'cyan': (118 / 255, 183 / 255, 178 / 255, 255 / 255),
	'brown': (156 / 255, 117 / 255, 95 / 255, 255 / 255),
	'gray': (186 / 255, 176 / 255, 172 / 255, 255 / 255),
}

PICK_TARGETS = {
	# 'red apple': None,
	# 'red strawberry': None,
	# 'yellow banana': None,
	# 'pink peach': None,
	# 'green pear': None,
	# 'orange orange': None,
	'blue block': None,
	'red block': None,
	'green block': None,
	'orange block': None,
	'yellow block': None,
	'purple block': None,
	'pink block': None,
	'cyan block': None,
	'brown block': None,
	'gray block': None,
}

PLACE_TARGETS = {
	# 'red apple': None,
	# 'red strawberry': None,
	# 'yellow banana': None,
	# 'pink peach': None,
	# 'green pear': None,
	# 'orange orange': None,

	'blue block': None,
	'red block': None,
	'green block': None,
	'orange block': None,
	'yellow block': None,
	'purple block': None,
	'pink block': None,
	'cyan block': None,
	'brown block': None,
	'gray block': None,

	'blue bowl': None,
	'red bowl': None,
	'green bowl': None,
	'orange bowl': None,
	'yellow bowl': None,
	'purple bowl': None,
	'pink bowl': None,
	'cyan bowl': None,
	'brown bowl': None,
	'gray bowl': None,

	'top left corner':     (-0.3 + 0.05, -0.2 - 0.05, 0),
	'top side':            (0,           -0.2 - 0.05, 0),
	'top right corner':    (0.3 - 0.05,  -0.2 - 0.05, 0),
	'left side':           (-0.3 + 0.05, -0.5,        0),
	'middle':              (0,           -0.5,        0),
	'right side':          (0.3 - 0.05,  -0.5,        0),
	'bottom left corner':  (-0.3 + 0.05, -0.8 + 0.05, 0),
	'bottom side':         (0,           -0.8 + 0.05, 0),
	'bottom right corner': (0.3 - 0.05,  -0.8 + 0.05, 0),
}

fixed_destination = {
	'top left corner':     (-0.3 + 0.05, -0.2 - 0.05, 0),
	'top side':            (0,           -0.2 - 0.05, 0),
	'top right corner':    (0.3 - 0.05,  -0.2 - 0.05, 0),
	'left side':           (-0.3 + 0.05, -0.5,        0),
	'middle':              (0,           -0.5,        0),
	'right side':          (0.3 - 0.05,  -0.5,        0),
	'bottom left corner':  (-0.3 + 0.05, -0.8 + 0.05, 0),
	'bottom side':         (0,           -0.8 + 0.05, 0),
	'bottom right corner': (0.3 - 0.05,  -0.8 + 0.05, 0),
}


instruction_form = [
	"Touch the {}",
	"Pick the {} up",
	"Move the {} to the {}",
	"Put the {} onto the {}",
	# "Push the {} to the {}",
]

action_state = {
	'Touch': '0',
	'Pick': '1',
	'Move': '2',
	'Put': '3',
	# 'Push': '4'
}
