from moviepy.video.io.VideoFileClip import VideoFileClip

def split_video(original_video_path, output_clip1_path, output_clip2_path, action_duration=5, transition_duration=2):
	"""
	Splits a video into two clips based on action and transition durations.

	Parameters:
		original_video_path (str): Path to the original video.
		output_clip1_path (str): Path to save the first clip.
		output_clip2_path (str): Path to save the second clip.
		action_duration (int): Duration of each action in seconds.
		transition_duration (int): Duration of the transition time in seconds.
	"""
	# Load the original video
	video = VideoFileClip(original_video_path)
	total_duration = video.duration

	# Calculate time ranges for each clip
	clip1_end = action_duration
	clip2_start = action_duration + transition_duration
	clip2_end = clip2_start + int(total_duration - clip1_end - transition_duration)
	print(clip1_end, clip2_start, clip2_end)
	# Ensure durations are within the video length
	# if clip2_end > total_duration:
	# 	raise ValueError("The specified durations exceed the video length.")

	# Extract and save the clips
	clip1 = video.subclip(0, clip1_end)
	clip1.write_videofile(output_clip1_path, codec="libx264", audio_codec="aac")

	clip2 = video.subclip(clip2_start, clip2_end)
	clip2.write_videofile(output_clip2_path, codec="libx264", audio_codec="aac")

	print(f"Clips saved successfully: {output_clip1_path}, {output_clip2_path}")

def main():
	args = parse_args()
	path = '/home/tuan/Downloads/long_video_to_clips'
	sequence_of_action = '/home/tuan/Downloads/long_video_to_clips/actions.txt'
	videos = sorted([os.path.join(path, video) for video in os.listdir(path) if video.endswith("mp4")],
					key=lambda x:int(x[-5]), reverse=False)
	with open(sequence_of_action, 'w') as file:
		actions = []
		for i, video in enumerate(videos):
			# build the model from a config file and a checkpoint file
			model = init_recognizer(args.config, args.checkpoint, device=args.device)  # device can be 'cuda:0'
			# test a single image
			result = inference_recognizer(model, video)
			predict_value = int(result.predict_results.item())
			truth_label = load_label_map(args.label_map)
			label = list(truth_label.values())[predict_value]+ "\n"
			actions.append(label)
			# cap = cv2.VideoCapture(args.video)
			# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			# out = cv2.VideoWriter(args.out_filename, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
			# 					  (width, height))
			# while cap.isOpened():
			# 	flag, ori_frame = cap.read()
			# 	if not flag:
			# 		break
			#
			# 	cv2.putText(ori_frame, label, (int(width / 12), int(height / 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
			# 				(0, 0, 0), 2, cv2.LINE_AA)
			# 	out.write(ori_frame)
		file.writelines(actions)

if __name__ == '__main__':
	main()
