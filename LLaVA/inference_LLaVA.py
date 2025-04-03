from transformers import AutoTokenizer
from LLaVA.llava.model import LlavaLlamaForCausalLM
import torch
import os
import requests
from PIL import Image
from io import BytesIO
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer
import warnings
import shutil

warnings.filterwarnings("ignore")

# Load model on CPU
model_path = "4bit/llava-v1.5-13b-3GB"
device = torch.device("cpu")
kwargs = {"device_map": "cpu"}
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Load vision model
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device="cpu")
image_processor = vision_tower.image_processor


def caption_image(image_file, prompt):
    # Load Image
    if image_file.startswith("http"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    disable_torch_init()

    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # Process image
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].float().to(device)

    # Create prompt
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()

    # Process tokens
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(
        device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Generate caption
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.to(device),
            images=image_tensor.to(device),
            do_sample=False, temperature=0.0,
            max_new_tokens=4, use_cache=True, stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    output = outputs.rsplit("</s>", 1)[0]
    return output


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is running
    base_dataset_dir = os.path.abspath(os.path.join(script_dir, "..", "Dataset", "test"))
    #
    # image_path1 = os.path.join(base_dataset_dir, "0.jpg")
    # image_path2 = os.path.join(base_dataset_dir, "1.jpg")

    prompt = (
        "Identify the main object in the image that is directly interacted with by a human hand. "
        "Describe it using exactly two words: first its color, then its object type. "
        "Format: 'the [color] [object]', e.g., 'the blue block'. "
        "Only use objects from this allowed list: "
        "ball, cup, block, book, pen, apple, toy, bottle, key, bowl, remote. "
    )

    responses = [caption_image(os.path.join(base_dataset_dir,image_path), prompt) for image_path in
                 sorted([image for image in os.listdir(base_dataset_dir)],
                        reverse=False, key=lambda x: int(x.split(".")[0]))]

    for desc in responses:
        print(desc + "\n")

    # folder_list = sorted(
    #     [f for f in os.listdir("Dataset/Best")],
    #     key=lambda x: int(x.split(".")[0])
    # )
    # for i, folder in enumerate(folder_list):
    #     folder_path = os.path.join("Dataset/Best", folder)
    #     image_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")], key=lambda x: int(x.split(".")[0]))
    #     for image in image_list:
    #         caption_image(image, prompt)
    #     description.append([caption_image(image, prompt)[0] for image in image_list])
    # with open(txt_path, "w") as file:
    #     file.write(description)
