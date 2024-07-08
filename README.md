# [StoryGen & StorySalon] Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models (CVPR 2024)

This repository contains a comprehensive pipeline for running the PyTorch implementation of StoryGen: https://arxiv.org/abs/2306.00973/.
The official implementation:
[Project Page](https://haoningwu3639.github.io/StoryGen_Webpage/)  $\cdot$ [Paper](https://arxiv.org/abs/2306.00973/) $\cdot$ [Dataset](https://huggingface.co/datasets/haoningwu/StorySalon) $\cdot$ [Checkpoint](https://huggingface.co/haoningwu/StoryGen)

# Overview

Before running StoryGen inference code on the StorySalon dataset, it is important to understand the structure of the dataset. The dataset consists of 2 parts: Ebooks and videos. Processed Ebook data can be downloaded directly through (https://huggingface.co/datasets/haoningwu/StorySalon) and has 6 subsets ("African", "Bloom", "Book", "Digital", "Literacy", "StoryWeaver"), but the video part of StorySalon dataset is not directly available. Refer to Section II for Video Data preparation and processing.

## I. Install the following requirements
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.12](https://pytorch.org/)
- xformers == 0.0.13
- diffusers == 0.13.1
- accelerate == 0.17.1
- transformers == 4.27.4

A suitable [conda](https://conda.io/) environment named `storygen` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate storygen
```

## II. Video Data Preparation
### Data Processing Pipeline
The data processing pipeline includes several necessary steps: 
- Step 1: Extract the keyframes and their corresponding subtitles;
- Step 2: Detect and remove duplicate frames;
- Step 3: Segment text, people, and headshots in images; and remove frames that only contain real people;
- Step 4:Inpaint the text, headshots and real hands in the frames according to the segmentation mask;
- (Optional) Use Caption model combined with subtitles to generate a description of each image.

The keyframes and their corresponding subtitles can be extracted via:
```
python ./data_process/extract.py
```

The duplicate frames can be detected and removed via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/dup_remove.py
```

The text, people and headshots can be segmented, and the frames that only contain real people are then removed via:
```
python ./data_process/yolov7/human_ocr_mask.py
```

The text, headshots and real hands in the frames can be inpainted with [SDM-Inpainting](https://github.com/CompVis/stable-diffusion), according to the segmentation mask via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/SDM/inpaint.py
```

Besides, we also provide the code to get story-level paired image-text samples.
We can align the subtitles with visual frames by using Dynamic Time Warping(DTW) algorithm via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/align.py
```

(Optional) You can use [TextBind](https://github.com/SihengLi99/TextBind) or [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4) to obtain the caption of each image via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/TextBind/main_caption.py
CUDA_VISIBLE_DEVICES=0 python ./data_process/MiniGPT-v2/main_caption.py
```

### Step 0: Installing youtube-dl command line tool
Metadata for the 'video' subset of the StorySalon dataset can be found in `./data/metadata.json`. It includes the id, name, url, duration and the keyframe list after filtering the videos.

The official github repo recommends to use [youtube-dl](https://github.com/yt-dlp/yt-dlp) via:
```
youtube-dl --write-auto-sub -o 'file\%(title)s.%(ext)s' -f 135 [url]
```
However, if errors occur, try
```
sudo pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"
```

Then download youtube videos according to urls provided in meta data 
```
if not os.path.exists('rawVideo'):
    os.mkdir('rawVideo')
%cd ./rawVideo

import json
import os

with open('./Image_inpainted/Video/metadata.json') as f:
    metadata = json.load(f)

for video_id, video_data in metadata.items():
    url = video_data['video_url'][0][0]
    os.makedirs(video_id, exist_ok=True)
    output_path = os.path.join(video_id, '%(title)s.%(ext)s')
    !youtube-dl --write-auto-sub -o '{output_path}' -f bestvideo+bestaudio/best {url}
```
The keyframes extracted with the following data processing pipeline (step 1) can be filtered according to the keyframe list provided in the metadata to avoid manual selection.


#### Data from Open-source Libraries
For the open-source PDF data, you can directly download the frames, corresponding masks, description and narrative from [StorySalon](https://huggingface.co/datasets/haoningwu/StorySalon).


## Training
Before training, please download pre-trained StableDiffusion-1.5 from [SDM](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) (including vae, scheduler, tokenizer and unet). Then, all the pre-trained checkpoints should be placed into the corresponding location in the folder `./ckpt/stable-diffusion-v1-5/`

For Stage 1, pre-train the self-attention layers in SDM for StyleTransfer via:
```
CUDA_VISIBLE_DEVICES=0 accelerate launch train_StorySalon_stage1.py
```

For Stage 2, train the Visual-Language Context Module via:

```
CUDA_VISIBLE_DEVICES=0 accelerate launch train_StorySalon_stage2.py
```

For replicating the experiments on MS-COCO, train via:

```
CUDA_VISIBLE_DEVICES=0 accelerate launch train_COCO.py
```

If you have multiple GPUs to accelerate the training process, you can use:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train_StorySalon_stage2.py
```

## Inference
```
CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py
```


## Citation
	@inproceedings{liu2024intelligent,
      title     = {Intelligent Grimm -- Open-ended Visual Storytelling via Latent Diffusion Models}, 
      author    = {Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, Weidi Xie},
      booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2024},
	}

## Acknowledgements
Many thanks to the code bases from [diffusers](https://github.com/huggingface/diffusers) and [SimpleSDM](https://github.com/haoningwu3639/SimpleSDM).

## Contact
If you have any questions, please feel free to contact haoningwu3639@gmail.com or liuchang666@sjtu.edu.cn.
