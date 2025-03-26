#  🌠 DriveGEN
This is the official project repository for *[DriveGEN: Generalized and Robust 3D Detection in Driving via Controllable Text-to-Image Diffusion Generation](https://arxiv.org/abs/2503.11122)* (CVPR 2025)

## Abstract
In autonomous driving, vision-centric 3D detection aims to identify 3D objects from images. However, high data collection costs and diverse real-world scenarios limit the scale of training data. Once distribution shifts occur between training and test data, existing methods often suffer from performance degradation, known as Out-of-Distribution (OOD) problems. To address this, controllable Text-to-Image (T2I) diffusion offers a potential solution for training data enhancement, which is required to generate diverse OOD scenarios with precise 3D object geometry. Nevertheless, existing controllable T2I approaches are restricted by the limited scale of training data or struggle to preserve all annotated 3D objects. In this paper, we present DriveGEN, a method designed to improve the robustness of 3D detectors in Driving via Training-Free Controllable Text-to-Image Diffusion Generation. Without extra diffusion model training, DriveGEN consistently preserves objects with precise 3D geometry across diverse OOD generations, consisting of 2 stages: 1) Self-Prototype Extraction: We empirically find that self-attention features are semantic-aware but require accurate region selection for 3D objects. Thus, we extract precise object features via layouts to capture 3D object geometry, termed self-prototypes. 2) Prototype-Guided Diffusion: To preserve objects across various OOD scenarios, we perform semantic-aware feature alignment and shallow feature alignment during denoising. Extensive experiments demonstrate the effectiveness of DriveGEN in improving 3D detection.

## Data Preparation

### Monocular 3D object detection
- 1️⃣ Download the KITTI dataset from the *[official website](https://www.cvlibs.net/datasets/kitti/)*
- 2️⃣ Download the splits (the ImageSets folder) from *[MonoTTA](https://github.com/Hongbin98/MonoTTA/tree/main/ImageSets)*

Then, 
```
mkdir data && cd data
ln -s /your_path_KITTI .
mv ./ImageSets ./your_path_KITTI
```

### Multi-view 3D object detection
- 1️⃣ Download the nuScenes dataset from the *[official website](https://www.nuscenes.org/)*
- 2️⃣ (Optional) Download the nuScenes-C dataset from the *[Robo3D](https://ldkong.com/Robo3D)* benchmark


You can also download all generated images on *[Hugging Face](https://huggingface.co/datasets/anthemlin/DriveGEN-datasets)* 🤗

## Installation
Build the conda environment via
```
conda env create -f environment.yml
conda activate driveGEN
pip install -r requirements.txt
```

## Usage
I’m currently busy with my lessons, but I will release the code as soon as possible.


## Citation
If our DriveGEN method is helpful in your research, please consider citing our paper:
```
@inproceedings{lin2025drivegen,
  title={DriveGEN: Generalized and Robust 3D Detection in Driving via Controllable Text-to-Image Diffusion Generation},
  author={Lin, Hongbin and Guo, Zilu and Zhang, Yifan and Niu, Shuaicheng and Li, Yafeng and Zhang, Ruimao and Cui, Shuguang and Li, Zhen},
  booktitle={CVPR},
  year={2025}
}
```

## Acknowledgment
The code is greatly inspired by (heavily from) the [FreeControl🔗](https://github.com/genforce/freecontrol).

## Correspondence 
Please contact Hongbin Lin by [linhongbinanthem@gmail.com] if you have any questions.  📬

