import os
from LLaVA.inference_LLaVA import caption_image

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is running
base_dataset_dir = os.path.abspath(os.path.join(script_dir, "Dataset", "test"))
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

responses = [caption_image(os.path.join(base_dataset_dir, image_path), prompt) for image_path in
             sorted([image for image in os.listdir(base_dataset_dir)],
                    reverse=False, key=lambda x: int(x.split(".")[0]))]

for desc in responses:
    print(desc + "\n")
