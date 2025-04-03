# from mmaction.apis import inference_recognizer, init_recognizer
#
# config_path = '/home/tuan/Documents/Code/mmaction2/pretrained_file_and_checkpoint/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb.py'
# checkpoint_path = '/home/tuan/Documents/Code/mmaction2/pretrained_file_and_checkpoint/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb_20230317-ec6696ad.pth' # can be a local path
# img_path = '/home/tuan/Documents/Code/mmaction2/demo/practical_video/picking_reg_cube.mp4'   # you can specify your own picture path
#
# # build the model from a config file and a checkpoint file
# model = init_recognizer(config_path, checkpoint_path, device="cpu")  # device can be 'cuda:0'
# # test a single image
# result = inference_recognizer(model, img_path)
# print(type(result))
# print(int(result.predict_results.item()))
# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import numpy as np
import torch
from mmaction.apis import inference_recognizer, init_recognizer

try:
	import moviepy.editor as mpy
except ImportError:
	raise ImportError('Please install moviepy to enable output file')


def load_label_map(file_path):
	"""Load Label Map.

	Args:
		file_path (str): The file path of label map.
	Returns:
		dict: The label map (int -> label name).
	"""
	with open(file_path, 'r') as file:
		lines = file.readlines()
	lines = {i: x.strip() for i, x in enumerate(lines)}
	return lines


def parse_args():
	parser = argparse.ArgumentParser(description='MMAction2 demo')
	parser.add_argument('--video', default='/home/tuan/Documents/Code/mmaction2/demo'
										   '/practical_video/picking_reg_cube.mp4', help='video file/url')
	parser.add_argument('--out_filename', default='output.mp4',help='output filename')
	parser.add_argument(
		'--config',
		default=('/home/tuan/Documents/Code/mmaction2/pretrained_file_and_checkpoint/'
				 'tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb.py'),
		help='action recognition model config file path')
	parser.add_argument(
		'--checkpoint',
		default=('/home/tuan/Documents/Code/mmaction2/pretrained_file_and_checkpoint/'
				 'tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb_20230317-ec6696ad.pth'),
		help='action recognition model checkpoint file/url')
	parser.add_argument(
		'--label_map',
		default='/home/tuan/Documents/Code/mmaction2/tools/data/sthv2/label_map.txt',
		help='label map file')
	parser.add_argument(
		'--device', type=str, default='cpu', help='CPU/CUDA device option')
	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	# build the model from a config file and a checkpoint file
	model = init_recognizer(args.config, args.checkpoint, device=args.device)  # device can be 'cuda:0'
	# test a single image
	result = inference_recognizer(model, args.video)
	predict_value = int(result.predict_results.item())
	truth_label = load_label_map(args.label_map)
	label = list(truth_label.values())[predict_value]

	cap = cv2.VideoCapture(args.video)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	out = cv2.VideoWriter(args.out_filename, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
						  (width, height))
	while cap.isOpened():
		flag, ori_frame = cap.read()
		if not flag:
			break

		cv2.putText(ori_frame, label, (int(width / 12), int(height / 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
					(0, 0, 0), 2, cv2.LINE_AA)
		out.write(ori_frame)


if __name__ == '__main__':
	main()
