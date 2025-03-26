import argparse
import os, random
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import yaml
from omegaconf import OmegaConf

from libs.utils.utils import merge_sweep_config, generate_hash_key
from libs.model import make_pipeline
from libs.model.module.scheduler import CustomDDIMScheduler

from PIL import Image


#### generate the text prompt via the object annotations
def generate_prompt_from_labels_KITTI(label_file_path, ood_type):
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
        prompt = f"A photo of {objects}, in the {ood_type}, best quality, extremely detailed."
        inverted_prompt = f"A photo of {objects}, with a simple background, best quality, extremely detailed."

        paired_categories = "".join([f"({cat}; {cat})" for cat in sorted(found_categories)])
        return prompt, paired_categories, inverted_prompt
    else:
        return "A photo in the {ood_type}, best quality, extremely detailed.", "", "A photo with a simple background, best quality, extremely detailed."

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


#### Conduct stage 1:self-prototype extraction for KITTI (Monocular 3D object detection)
def main(args):
    #### load the stable diffusion model
    model_config_path = 'config/model_config.yaml'
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    models_info = model_config["checkpoints"]
    if args.sd_version not in models_info.keys():
        raise ValueError(f"Model {args.sd_version} not found in the model list: {list(models_info.keys())}.")
    model_ckpt_list = models_info[args.sd_version]

    if args.model_name not in model_ckpt_list.keys():
        raise ValueError(
            f"Stable Diffusion version {args.model_name} not found in the model {args.sd_version} list: {list(model_ckpt_list.keys())}.")
    model_path = model_ckpt_list[args.model_name]['path']

    config_path = 'config/KITTI.yaml'
    # base_config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    # base_config = OmegaConf.create(base_config)
    if 'XL' in args.sd_version:
        pipeline_name = "SDXLPipeline"
    else:
        pipeline_name = "SDPipeline"

    pipeline = make_pipeline(pipeline_name,
                             model_path,
                             torch_dtype=torch.float16
                             ).to('cuda')
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_sequential_cpu_offload()
    pipeline.scheduler = CustomDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # #### load data info. for all KITTI
    # file_path = 'ImageSets/train.txt'   # please remember to download the data splits
    # with open(file_path, 'r') as file:
    #     lines_list = file.read().splitlines()
    #     random.shuffle(lines_list)
    #     img_idxes = lines_list

    #### Just for one specific image       
    img_idxes = ['000032']

    proto_path = "temp_data/KITTI_protos"
    latent_path = "temp_data/KITTI_latents"
    image_path = "temp_data/KITTI_res"
    
    processor = lambda x: Image.open(x).convert("RGB") if type(x) == str else x
    base_config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)    
    for idx in img_idxes:
        if not os.path.exists(f"{proto_path}/{idx}.pt"):
            condition_image = Image.open(os.path.join(args.data_root, 'image_2', idx + '.png'))    # load the input image
            prompt, paired_objs, inversion_prompt = generate_prompt_from_labels_KITTI(os.path.join(args.data_root, 'label_2', idx + '.txt'), args.ood_type)    # load the annotations
            # print(prompt, paired_objs, inversion_prompt)

            update_parameter = {
                'data--inversion--prompt': inversion_prompt,
                'data--inversion--fixed_size': [base_config['sd_config']['W'], base_config['sd_config']['H']],  # must be diveded by 8
            }            
            config = merge_sweep_config(base_config=base_config, update=update_parameter)
            config = OmegaConf.create(config)

            # inversion config
            inversion_config = config.data.inversion
            inversion_config.target_folder = latent_path    # dir to save the inverted latent

            # Process the condition image
            img = processor(condition_image)
            img = img.resize((config['sd_config']['W'], config['sd_config']['H']))

            # Apply inversion methods
            real_image_latents = []
            real_image_latents.append(pipeline.invert(img=img, inversion_config=inversion_config))

            pipeline.sample_semantic_bases_with_realimages(
                prompt=inversion_prompt,
                negative_prompt=config['sd_config']['negative_prompt'],
                generator=g,
                num_inference_steps=config['sd_config']['steps'],
                height=config['sd_config']['H'],
                width=config['sd_config']['W'],
                num_images_per_prompt=1,
                num_batch=1,
                config=config,
                num_save_basis=args.prototype_nums,
                num_save_steps=args.save_steps,
                real_images_ddim=real_image_latents
            )
            os.makedirs(proto_path, exist_ok=True)
            pca_info = pipeline.pca_info 
            torch.save(pca_info, f"{proto_path}/{idx}.pt")
        else:
            print(f"{proto_path}/{idx}.pt")

    #### Stage 2: image generatino
    for idx in img_idxes:
        if os.path.exists(os.path.join(image_path, f"{idx}.png")): continue
        
        condition_image = Image.open(os.path.join(args.data_root, 'image_2', idx + '.png'))    # load the input image
        prompt, paired_objs, inversion_prompt = generate_prompt_from_labels_KITTI(os.path.join(args.data_root, 'label_2', idx + '.txt'), args.ood_type)    # load the annotations

        original_size = condition_image.size  # record the original image size 
        bboxes_dict = read_and_scale_bboxes_KITTI(os.path.join(args.data_root, 'label_2', idx + '.txt'), condition_image, base_config['sd_config']['W'], base_config['sd_config']['H'])  # load the image layout

        generation_parameter = {
            'sd_config--guidance_scale': base_config['sd_config']['guidance_scale'],
            'sd_config--steps': base_config['sd_config']['steps'],
            'sd_config--seed': args.seed,
            'sd_config--prompt': prompt,
            'sd_config--negative_prompt': base_config['sd_config']['negative_prompt'],
            'sd_config--obj_pairs': str(paired_objs),
            'sd_config--pca_paths': [os.path.join(proto_path, idx + '.pt')],
            'data--inversion--prompt': inversion_prompt,
            'data--inversion--fixed_size': [base_config['sd_config']['W'], base_config['sd_config']['H']],
            'guidance--pca_guidance--end_step': int(0.6 * base_config['sd_config']['steps']),
            'guidance--pca_guidance--weight': 600,
            'guidance--pca_guidance--structure_guidance--n_components': args.prototype_nums,
            'guidance--pca_guidance--structure_guidance--normalize': True,
            'guidance--pca_guidance--structure_guidance--mask_tr': 0.3,
            'guidance--pca_guidance--structure_guidance--penalty_factor': 10,
            'guidance--pca_guidance--warm_up--apply': True,
            'guidance--pca_guidance--warm_up--end_step': int(0.05 * base_config['sd_config']['steps']),
            'guidance--cross_attn--end_step': int(0.6 * base_config['sd_config']['steps']),
            'guidance--cross_attn--weight': 0,
        }
        input_config = generation_parameter
        config = merge_sweep_config(base_config=base_config, update=input_config)
        config = OmegaConf.create(config)

        inversion_config = config.data.inversion
        inversion_config.target_folder = latent_path    # dir to save the inverted latent
        
        img = processor(condition_image)
        condition_image_latents = pipeline.invert(img=img, inversion_config=inversion_config)
        inverted_data = {"condition_input": [condition_image_latents]}

        g = torch.Generator()
        g.manual_seed(config.sd_config.seed)

        # generate the img
        img_list = pipeline.ddim_inverted_2_image(
            prompt=config.sd_config.prompt,
            negative_prompt=config.sd_config.negative_prompt,
            num_inference_steps=config.sd_config.steps,
            generator=g,
            config=config,
            inverted_data=inverted_data,
            bboxes_dict=bboxes_dict)[0]

        # save the output image
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        resized_img = img_list[0].resize(original_size, Image.LANCZOS)
        resized_img.save(os.path.join(image_path, f"{idx}.png"))
        print(f"Images saved to {image_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images using the provided configuration.')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
    parser.add_argument('--sd_version', type=str, default='1.5', help='Stable Diffusion version.')
    parser.add_argument('--model_name', type=str, default="naive", help='Model name.')
    parser.add_argument('--data_root', type=str, default='data/KITTI', help='Path of KITTI.')
    parser.add_argument('--ood_type', type=str, default="sniw", help='Select the corruption.')
    parser.add_argument('--prototype_nums', type=int, default=32, help='Number of self-prototypes to save.')
    parser.add_argument('--save_steps', type=int, default=120, help='Number of steps to save the self-prototypes.')
    args = parser.parse_args()

    main(args)
