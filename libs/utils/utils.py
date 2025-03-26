import hashlib
import io
import os
import re

import PIL.Image
import PIL.ImageOps
import numpy as np
import requests
import torch
from PIL import Image
from typing import Callable, Union

def set_nested_item(dataDict, mapList, value):
    """Set item in nested dictionary"""
    """
    Example: the mapList contains the name of each key ['injection','self-attn']
            this method will change the content in dataDict['injection']['self-attn'] with value

    """
    for k in mapList[:-1]:
        dataDict = dataDict[k]
    dataDict[mapList[-1]] = value


def merge_sweep_config(base_config, update):
    """Merge the updated parameters into the base config"""

    if base_config is None:
        raise ValueError("Base config is None")
    if update is None:
        raise ValueError("Update config is None")
    for key in update.keys():
        map_list = key.split("--")
        set_nested_item(base_config, map_list, update[key])
    return base_config


# Adapt from https://github.com/castorini/daam
def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0):
    merge_idxs = []
    tokens = tokenizer.tokenize(prompt.lower())
    if word_idx is None:
        word = word.lower()
        search_tokens = tokenizer.tokenize(word)
        start_indices = [x + offset_idx for x in range(len(tokens)) if
                         tokens[x:x + len(search_tokens)] == search_tokens]
        for indice in start_indices:
            merge_idxs += [i + indice for i in range(0, len(search_tokens))]
        if not merge_idxs:
            raise Exception(f'Search word {word} not found in prompt!')
    else:
        merge_idxs.append(word_idx)

    return [x + 1 for x in merge_idxs], word_idx  # Offset by 1.


def extract_data(input_string: str) -> list:
    print("input_string:", input_string)
    """
    Extract data from a string pattern where contents in () are separated by ;
    The first item in each () is considered as 'ref' and the rest as 'gen'.

    Args:
    - input_string (str): The input string pattern.

    Returns:
    - list: A list of dictionaries containing 'ref' and 'gen'.
    """
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, input_string)

    data = []
    for match in matches:
        parts = [x.strip() for x in match.split(';')]
        ref = parts[0].strip()
        gen = parts[1].strip()
        data.append({'ref': ref, 'gen': gen})

    return data


def generate_hash_key(image, prompt=""):
    """
    Generate a hash key for the given image and prompt.
    """
    byte_array = io.BytesIO()
    image.save(byte_array, format='JPEG')

    # Get byte data
    image_byte_data = byte_array.getvalue()

    # Combine image byte data and prompt byte data
    combined_data = image_byte_data + prompt.encode('utf-8')

    sha256 = hashlib.sha256()
    sha256.update(combined_data)
    return sha256.hexdigest()


def save_data(data, folder_path, key):
    """
    Save data to a file, using key as the file name
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{key}.pt")

    torch.save(data, file_path)


def get_data(folder_path, key):
    """
    Get data from a file, using key as the file name
    :param folder_path:
    :param key:
    :return:
    """

    file_path = os.path.join(folder_path, f"{key}.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)
    else:
        return None


def PILtoTensor(data: Image.Image) -> torch.Tensor:
    return torch.tensor(np.array(data)).permute(2, 0, 1).unsqueeze(0).float()


def TensorToPIL(data: torch.Tensor) -> Image.Image:
    return Image.fromarray(data.squeeze().permute(1, 2, 0).numpy().astype(np.uint8))

# Adapt from https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/utils/loading_utils.py#L9
def load_image(
        image: Union[str, PIL.Image.Image], convert_method: Callable[[PIL.Image.Image], PIL.Image.Image] = None
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], optional):
            A conversion method to apply to the image after loading it.
            When set to `None` the image will be converted "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image


#### generate the text prompt via the object annotations
def generate_inversion_prompt_from_labels_KITTI(label_file_path):
    # define the valid categories
    valid_categories = {"Car", "Pedestrian", "Cyclist"}
    found_categories = set()

    # load the label file
    with open(label_file_path, 'r') as file:
        for line in file:
            # get the class name
            parts = line.split()
            if parts[0] in valid_categories:
                found_categories.add(parts[0])
    
    if found_categories:
        # combine all valid categories
        objects = ", ".join(sorted(found_categories))
        prompt = f"A photo of {objects}, with a simple background, best quality, extremely detailed."
        return prompt
    else:
        return "A photo with a simple background, best quality, extremely detailed."

#### load the annotations of bboxes in the KITTI format
def read_and_scale_bboxes_KITTI(label_file_path, condition_image, w_resized, h_resized):
    bbox_dict = {"Car": [], "Pedestrian": [], "Cyclist": []}

    # get the size of the input image
    img_width, img_height = condition_image.size
    
    # calculate the scaling factor
    scale_x = w_resized / img_width
    scale_y = h_resized / img_height
    
    with open(label_file_path, 'r') as file:
        for line in file:
            parts = line.split()
            category = parts[0]
            if category in bbox_dict:
                x1, y1, x2, y2 = map(float, parts[4:8])

                # scaling the annotations
                x1_resized = int(x1 * scale_x)
                y1_resized = int(y1 * scale_y)
                x2_resized = int(x2 * scale_x)
                y2_resized = int(y2 * scale_y)
                
                bbox_dict[category].append([x1_resized, y1_resized, x2_resized, y2_resized])
    
    return bbox_dict


#### generate text prompts for image generation
def get_text_prompts_generation(label_file_path, ood_type):
    """
    Generate a prompt string based on categories found in a label file.
    """
    valid_categories = {"Car", "Pedestrian", "Cyclist"}
    found_categories = set()

    # Read the label file line by line
    with open(label_file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if parts[0] in valid_categories:
                found_categories.add(parts[0])
    
    # If any valid category was found, build the prompts
    if found_categories:
        objects = ", ".join(sorted(found_categories))
        prompt = f"A photo of {objects}, in the {ood_type}, best quality, extremely detailed."
        inverted_prompt = f"A photo of {objects}, with a simple background, best quality, extremely detailed."
        paired_categories = "".join([f"({cat}; {cat})" for cat in sorted(found_categories)])
        return prompt, paired_categories, inverted_prompt
    
    # Otherwise, return generic prompts
    return (
        "A photo in the {ood_type}, best quality, extremely detailed.",
        "",
        "A photo with a simple background, best quality, extremely detailed."
    )
