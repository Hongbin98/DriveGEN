import argparse
import os

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import yaml
from omegaconf import OmegaConf

from libs.utils.utils import merge_sweep_config, generate_hash_key, load_json_data
from libs.model import make_pipeline
from libs.model.module.scheduler import CustomDDIMScheduler

from PIL import Image


# #### simplify the class name of nuscenes
# NameMapping = {
#     'movable_object.barrier': 'barrier',
#     'vehicle.bicycle': 'bicycle',
#     'vehicle.bus.bendy': 'bus',
#     'vehicle.bus.rigid': 'bus',
#     'vehicle.car': 'car',
#     'vehicle.construction': 'construction_vehicle',
#     'vehicle.motorcycle': 'motorcycle',
#     'human.pedestrian.adult': 'pedestrian',
#     'human.pedestrian.child': 'pedestrian',
#     'human.pedestrian.construction_worker': 'pedestrian',
#     'human.pedestrian.police_officer': 'pedestrian',
#     'movable_object.trafficcone': 'traffic_cone',
#     'vehicle.trailer': 'trailer',
#     'vehicle.truck': 'truck'
# }


#### Conduct stage 1:self-prototype extraction for nuscenes (Multi-view 3D object detection)
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


    if args.model_name not in model_ckpt_list.keys():
        raise ValueError(
            f"Stable Diffusion version {args.model_name} not found in the model {args.sd_version} list: {list(model_ckpt_list.keys())}.")
    model_path = model_ckpt_list[args.model_name]['path']

    config_path = 'config/nuscenes.yaml'
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    config = OmegaConf.create(config)
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

    data = load_json_data('organized_2d_bboxes.json')
    proto_path = "temp_data_{}/nus_protos".format(args.sd_version)
    latent_path = "temp_data_{}/nus_latents".format(args.sd_version)    
    for filename, bboxes in data.items():
        if not os.path.exists(f"{proto_path}/{filename.replace('/', '_').replace('samples_', '', 1)}.pt"):    
            tmp_bboxes = []
            
            #### open the image
            image_path = os.path.join(args.data_root, filename)
            condition_image = Image.open(image_path)        

            #### generate the prompt with cls names
            filename = filename.replace('/', '_').replace('samples_', '', 1)
            found_categories = set()
            for bbox in bboxes:
                category_name = bbox['category_name']
                x1, y1, x2, y2 = map(int, bbox['bbox_corners'])
                tmp_bboxes.append([x1, y1, x2, y2])
                found_categories.add(category_name)

            if found_categories:
                objects = ", ".join(sorted(found_categories))
                prompt = f"A photo of {objects}, with a simple background, best quality, extremely detailed."
            else:
                prompt =  "A photo with a simple background, best quality, extremely detailed."

            config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
            update_parameter = {
                'data--inversion--prompt': prompt,
                'data--inversion--fixed_size': [config['sd_config']['W'], config['sd_config']['H']],  # must be diveded by 8
            }            
            config = merge_sweep_config(base_config=config, update=update_parameter)
            config = OmegaConf.create(config)

            # inversion config
            inversion_config = config.data.inversion
            inversion_config.target_folder = latent_path    # dir to save the inverted latent

            # Processor the condition image
            processor = lambda x: Image.open(x).convert("RGB") if type(x) == str else x
            img = processor(condition_image)
            img = img.resize((config['sd_config']['W'], config['sd_config']['H']))

            # Apply inversion methods
            real_image_latents = []
            real_image_latents.append(pipeline.invert(img=img, inversion_config=inversion_config))

            pipeline.self_prototype_extraction(
                prompt=prompt,
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
                real_image_latents=real_image_latents
            )
            os.makedirs(proto_path, exist_ok=True)
            self_prototypes = pipeline.self_prototypes 
            torch.save(self_prototypes, f"{proto_path}/{filename}.pt")
        else:
            print(f"{proto_path}/{filename.replace('/', '_').replace('samples_', '', 1)}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate self-prototypes using the provided configuration.')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
    parser.add_argument('--sd_version', type=str, default='2.1_base', help='Stable Diffusion version.')
    parser.add_argument('--model_name', type=str, default="naive", help='Model name.')
    parser.add_argument('--data_root', type=str, default='data/nuScenes', help='Path of KITTI.')
    parser.add_argument('--prototype_nums', type=int, default=16, help='Number of self-prototypes to save.')
    parser.add_argument('--save_steps', type=int, default=120, help='Number of steps to save the self-prototypes.')
    args = parser.parse_args()

    main(args)
