from ObjectDetection.vild import VILD, build_scene_description
from PromptEngineering.prompt_engineering import make_options
from Video_keyframe_detector.KeyFrameDetector.key_frame_detector import keyframeDetection
from utils.PromptEngineering_scoring import gpt3_5_turbo_scoring
from utils.Scene_scoring import affordance_scoring
import argparse
import numpy as np
import os
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import torch
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer

# pre-defined constant variables
PICK_TARGETS = {
    "red block": None,
    "blue block": None,
    "green block": None,
    "yellow block": None,
}

COLORS = {
    "blue": (78 / 255, 121 / 255, 167 / 255, 255 / 255),
    "red": (255 / 255, 87 / 255, 89 / 255, 255 / 255),
    "green": (89 / 255, 169 / 255, 79 / 255, 255 / 255),
    "yellow": (237 / 255, 201 / 255, 72 / 255, 255 / 255),
}

MOVE_TARGETS = {
"top": None,
"bottom": None,
"front": None,
"left": None,
"right": None,
"back": None,

}

PLACE_TARGETS = {
    "blue block": None,
    "red block": None,
    "green block": None,
    "yellow block": None,

    "blue bowl": None,
    "red bowl": None,
    "green bowl": None,
    "yellow bowl": None,

    "top left corner": (-0.3 + 0.05, -0.2 - 0.05, 0),
    "top right corner": (0.3 - 0.05, -0.2 - 0.05, 0),
    "middle": (0, -0.5, 0),
    "bottom left corner": (-0.3 + 0.05, -0.8 + 0.05, 0),
    "bottom right corner": (0.3 - 0.05, -0.8 + 0.05, 0),
}

SENTENCES = {
"Pick the {PICK_TARGETS} and place it on the {PLACE_TARGETS}.",
"Move the manipulator to the {MOVE_TARGETS} position."
}

GPT3_5_TURBO_CONTEXT = """
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue block, top left corner)
robot.pick_and_place(red block, top left corner)
robot.pick_and_place(yellow block, top left corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green block, blue block)
robot.pick_and_place(red block, green block)
done()

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue block, blue bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green block, green bowl)
robot.pick_and_place(red block, red bowl)
robot.pick_and_place(yellow block, yellow bowl)
done()

object = []
# move the manipulator to the top position.
robot.move(top)
done()
"""

CATEGORY_NAMES = ['blue block',
                      'gray block',
                      'cyan block',
                      'white block',
                      'black block',

                      'white bowl',
                      'gray bowl',
                      ]

model_path = "4bit/llava-v1.5-13b-3GB"
kwargs = {"device_map": "auto"}
kwargs["load_in_4bit"] = True
kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4"
)

model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device="cuda")
image_processor = vision_tower.image_processor

def caption_image(image_file, prompt):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                    max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit("</s>", 1)[0]
    return image, output

def normalize_scores(scores):
    max_score = max(scores.values())
    normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
    return normed_scores


def prompt_creation(env_description, gpt3_5_turbo_context, raw_input):
    options = make_options(env_description, MOVE_TARGETS, PICK_TARGETS, PLACE_TARGETS)
    use_environment_description = True
    gpt3_5_turbo_context_lines = gpt3_5_turbo_context.split("\n")
    gpt3_5_turbo_context_lines_keep = []

    for gpt3_5_turbo_context_line in gpt3_5_turbo_context_lines:
        if "objects =" in gpt3_5_turbo_context_line and not use_environment_description:
            continue
        gpt3_5_turbo_context_lines_keep.append(gpt3_5_turbo_context_line)

    gpt3_5_turbo_context = "\n".join(gpt3_5_turbo_context_lines_keep)

    gpt3_5_turbo_prompt = gpt3_5_turbo_context
    if use_environment_description:
        gpt3_5_turbo_prompt += "\n" + env_description
    gpt3_5_turbo_prompt += "\n# " + raw_input + "\n"

    return gpt3_5_turbo_prompt, options

def main(args):
    category_name_string = ";".join(CATEGORY_NAMES)
    all_llm_scores = []
    all_affordance_scores = []
    all_combined_scores = []
    max_tasks = 5
    selected_task = ""
    steps_text = []
    termination_string = "done()"

    # Extra prompt engineering: swap A with B for every (A, B) in list.
    prompt_swaps = [('block', 'cube')]

    ''' vild params'''
    max_boxes_to_draw = 8  # @param {type:"integer"}
    nms_threshold = 0.4  # @param {type:"slider", min:0, max:0.9, step:0.05}
    min_rpn_score_thresh = 0.4  # @param {type:"slider", min:0, max:1, step:0.01}
    min_box_area = 10  # @param {type:"slider", min:0, max:10000, step:1.0}
    max_box_area = 3000  # @param {type:"slider", min:0, max:10000, step:1.0}
    vild_params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area

    ''' Main '''
    keyframeDetection(args.source, args.dest, float(args.Thres))
    key_frame_folder_path = "../Key_frames/keyFrames"
    key_frames = [os.path.join(key_frame_folder_path, image) for image in os.listdir(key_frame_folder_path)]
    # print(key_frames)
    query = "Describe only the action in one sentence without mentioning the object"
    descriptions = [caption_image(image, query)[1] for image in key_frames[:7]]
    print(descriptions)
    # descriptions = ["move all the blocks to the top left corner" for _ in len(key_frames)]
    scenes = [VILD(image_path=image, category_name_string=category_name_string, params=vild_params, plot_on=False, prompt_swaps=prompt_swaps) for
              image in key_frames[:7]]
    print(scenes)
    obj_and_desc = zip(scenes, descriptions)

    for found_objects, description in obj_and_desc:
        selected_task = ""
        step_text = []
        num_tasks = 0
        env_description = build_scene_description(found_objects)
        gpt3_5_turbo_prompt, options = prompt_creation(env_description, GPT3_5_TURBO_CONTEXT, description)

        scene_score = affordance_scoring(options, found_objects, verbose=False, block_name="box", bowl_name="circle",
                           termination_string="done()")

        while not selected_task == termination_string:
            num_tasks += 1
            if num_tasks > max_tasks:
                selected_task = "done()"
                step_text.append(selected_task)
                break

            # gpt-3.5-turbo compute prompt engineering score
            llm_scores, _ = gpt3_5_turbo_scoring(gpt3_5_turbo_prompt, options, engine=args.engine, limit_num_options=None,
                                            option_start="\n", verbose=False,
                                            print_tokens=False)
            combined_scores = {option: np.exp(llm_scores[option]) * scene_score[option] for option in options}
            combined_scores = normalize_scores(combined_scores)
            selected_task = max(combined_scores, key=combined_scores.get)
            step_text.append(selected_task)
            print(num_tasks, "Selecting: ", selected_task)
            gpt3_5_turbo_prompt += selected_task + "\n"

            all_llm_scores.append(llm_scores)
            all_affordance_scores.append(scene_score)
            all_combined_scores.append(combined_scores)

        steps_text.append(step_text)
    print(steps_text)
    return steps_text

def argumentparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source', help='source file', required=True, default="./block_grapping.mp4")
    parser.add_argument('-d', '--dest', help='destination folder', required=True, default="VIDRoBo/Key_frames")
    parser.add_argument('-t', '--Thres', help='Threshold of the image difference', default=0.3)
    parser.add_argument('-e', '--engine', help='GPT engine', default="gpt-3.5-turbo-1106")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = argumentparse()
    main(args)
