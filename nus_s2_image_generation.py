import os.path
import argparse
from typing import Dict, List

import numpy as np
import torch
import yaml
from PIL import Image
from omegaconf import OmegaConf

from libs.utils.utils import merge_sweep_config, read_and_scale_bboxes_nus, load_json_data
from libs.model import make_pipeline
from libs.model.module.scheduler import CustomDDIMScheduler
from libs.utils.controlnet_processor import make_processor

import random


def DriveGEN_generate(scale, ddim_steps, sd_version,
                        model_ckpt, guidance_steps, guidance_prototypes,
                        guidance_weight, guidance_normalized,
                        masked_tr, guidance_penalty_factor, warm_up_step, negative_prompt, seed, data_root):
    model_ckpt = 'naive'    
    processor = lambda x: Image.open(x).convert("RGB") if type(x) == str else x

    # load the pipeline
    model_path = model_dict[sd_version][model_ckpt]['path']
    pipeline_name = "SDXLPipeline" if 'XL' in sd_version else "SDPipeline"
    pipeline = make_pipeline(pipeline_name, model_path, torch_dtype=torch.float16).to('cuda')
    pipeline.scheduler = CustomDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

    w_resized = 1600
    h_resized = 896 

    ood_type = 'snow'
    proto_path = "temp_data_{}/nus_protos".format(sd_version)
    latent_path = "temp_data_{}/nus_latents".format(sd_version)    
    save_path = "temp_data_{}/nus_res".format(sd_version)
    save_path = os.path.join(save_path, ood_type)


    # load the basic config
    config_path = 'config/nuscenes.yaml'
    base_config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    data = load_json_data('organized_2d_bboxes.json')

    for filename, bboxes in data.items():
        tmp_bboxes = []

        #### open the image
        image_path = os.path.join(data_root, filename)
        condition_image = Image.open(image_path)

        filename = filename.replace('/', '_').replace('samples_', '', 1)

        #### generate the prompt with cls names
        found_categories = set()
        for bbox in bboxes:
            category_name = bbox['category_name']
            x1, y1, x2, y2 = map(int, bbox['bbox_corners'])

            tmp_bboxes.append([x1, y1, x2, y2])
            found_categories.add(category_name)

        if found_categories:
            objects = ", ".join(sorted(found_categories))
            inversion_prompt = f"A photo of {objects}, with a simple background, best quality, extremely detailed."
            prompt = f"A photo of {objects}, in the {ood_type}, best quality, extremely detailed."
        else:
            inversion_prompt =  "A photo with a simple background, best quality, extremely detailed."
            prompt = f"A photo of {objects}, in the {ood_type}, best quality, extremely detailed."


        paired_objs = "".join([f"({cat}; {cat})" for cat in sorted(found_categories)])
        print(filename, prompt, paired_objs, inversion_prompt)  

        original_size = condition_image.size
        bboxes_dict = read_and_scale_bboxes_nus(bboxes, condition_image)

        # configs
        gradio_update_parameter = {
            'sd_config--guidance_scale': scale,
            'sd_config--steps': ddim_steps,
            'sd_config--seed': seed,
            'sd_config--prompt': prompt,
            'sd_config--negative_prompt': negative_prompt,
            'sd_config--obj_pairs': str(paired_objs),
            'sd_config--prototype_path': [os.path.join(proto_path, filename + '.pt')],
            'data--inversion--prompt': inversion_prompt,
            'data--inversion--fixed_size': [w_resized, h_resized],
            'guidance--guidance--end_step': int(guidance_steps * ddim_steps),
            'guidance--guidance--weight': guidance_weight,
            'guidance--guidance--structure_guidance--n_components': guidance_prototypes,
            'guidance--guidance--structure_guidance--normalize': bool(guidance_normalized),
            'guidance--guidance--structure_guidance--mask_tr': masked_tr,
            'guidance--guidance--structure_guidance--penalty_factor': guidance_penalty_factor,
            'guidance--guidance--warm_up--apply': True if warm_up_step > 0 else False,
            'guidance--guidance--warm_up--end_step': int(warm_up_step * ddim_steps),
            'guidance--cross_attn--end_step': int(guidance_steps * ddim_steps),
            'guidance--cross_attn--weight': 0,
        }
        input_config = gradio_update_parameter
        config = merge_sweep_config(base_config=base_config, update=input_config)
        config = OmegaConf.create(config)

        # inversion config
        inversion_config = config.data.inversion
        inversion_config.target_folder = latent_path

        # Processor the condition image
        img = processor(condition_image)
        condition_image_latents = pipeline.invert(img=img, inversion_config=inversion_config)
        inverted_data = {"condition_input": [condition_image_latents]}

        # set the random seed
        g = torch.Generator()
        g.manual_seed(config.sd_config.seed)

        # generate the image
        img_list = pipeline.prototype_guided_image_generation(
            prompt=config.sd_config.prompt,
            negative_prompt=config.sd_config.negative_prompt,
            num_inference_steps=config.sd_config.steps,
            generator=g,
            config=config,
            inverted_data=inverted_data,
            bboxes_dict=bboxes_dict)[0]

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        resized_img = img_list[0].resize(original_size, Image.LANCZOS)
        resized_img.save(os.path.join(save_path, f"{filename}"))
        print(f"Images saved to {save_path}")


def load_ckpt(config_path='config/model_config.yaml'):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")

    with open(config_path, 'r') as f:
        gradio_config = yaml.safe_load(f)

    models: Dict = gradio_config['checkpoints']
    pca_basis_dict: Dict = dict()

    for model_version in list(models.keys()):
        for model_name in list(models[model_version].keys()):
            if "naive" not in model_name and not os.path.isfile(models[model_version][model_name]["path"]):
                models[model_version].pop(model_name)

    return models, pca_basis_dict


if __name__ == '__main__':
    # load the model info
    model_dict, _ = load_ckpt()

    parser = argparse.ArgumentParser(description='Generate images using the provided configuration.')
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="Number of DDIM sampling steps."
    )
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1_base",
        help="Stable Diffusion model version."
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="radio"
    )
    parser.add_argument(
        "--guidance_steps",
        type=float,
        default=0.6,
        help="Percentage of sampling steps during which the guidance is applied."
    )
    parser.add_argument(
        "--guidance_prototypes",
        type=int,
        default=16,
        help="Number of self-prototypes for guidance."
    )
    parser.add_argument(
        "--guidance_weight",
        type=float,
        default=600,
        help="Overall strength for PCA guidance."
    )
    parser.add_argument(
        "--guidance_normalized",
        type=bool,
        default=True,
        help="Enable or disable normalization for PCA guidance."
    )
    parser.add_argument(
        "--masked_tr",
        type=float,
        default=0.3,
        help="Pre-defined threshold for region masking."
    )
    parser.add_argument(
        "--guidance_penalty_factor",
        type=float,
        default=10,
        help="Penalty factor for guiding the PCA direction."
    )
    parser.add_argument(
        "--warm_up_step",
        type=float,
        default=0.05,
        help="Initial portion of steps used for PCA warm-up."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, irregular car, incomplete car",
        help="Comma-separated list of negative prompts."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        '--data_root', 
        type=str, 
        default='data/nuScenes', 
        help='Path of nuscenes.')

    args = parser.parse_args()

    # Generate images via DriveGEN
    DriveGEN_generate(
        scale=args.scale,
        ddim_steps=args.ddim_steps,
        sd_version=args.sd_version,
        model_ckpt=args.model_ckpt,
        guidance_steps=args.guidance_steps,
        guidance_prototypes=args.guidance_prototypes,
        guidance_weight=args.guidance_weight,
        guidance_normalized=args.guidance_normalized,
        masked_tr=args.masked_tr,
        guidance_penalty_factor=args.guidance_penalty_factor,
        warm_up_step=args.warm_up_step,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        data_root=args.data_root
    )
