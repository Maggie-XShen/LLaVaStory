#Modified -- output named as: Output_StorySalon_Ebooks/categoryName/storyID/storyID_frameID
import os
from typing import Optional
from torchvision import transforms
from PIL import Image
from typing import List, Optional, Union

import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, CLIPTextModel

from utils.util import get_time_string
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StableDiffusionPipeline

logger = get_logger(__name__)

def test(
    pretrained_model_path: str,
    logdir: str,
    prompt: str,
    ref_prompt: Union[str, List[str]],
    ref_image: Union[str, List[str]],
    num_inference_steps: int = 40,
    guidance_scale: float = 7.0,
    image_guidance_scale: float = 3.5,
    num_sample_per_prompt: int = 1,
    stage: str = "multi-image-condition", # ["multi-image-condition", "auto-regressive", "no"]
    mixed_precision: Optional[str] = "fp16" ,
    frame_number: str = ""
):
    # time_string = get_time_string()
    logdir = os.path.join(logdir, f"{frame_number}")  # Append frame number to logdir
    # story_output_dir = os.path.dirname(logdir)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    accelerator = Accelerator(mixed_precision=mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed" f" correctly and a GPU is available: {e}"
            )
    unet, pipeline = accelerator.prepare(unet, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("StoryGen")

    vae.eval()
    text_encoder.eval()
    unet.eval()
    
    ref_images= []
    for id in ref_image:
        r_image = Image.open(id).convert('RGB').resize((512, 512))
        r_image = transforms.ToTensor()(r_image)
        ref_images.append(np.ascontiguousarray(r_image))
    ref_images = torch.from_numpy(np.ascontiguousarray(ref_images)).float()
    for ref_image in ref_images:
        ref_image = ref_image * 2. - 1.
    ref_images = ref_images.unsqueeze(0)

    sample_seeds = torch.randint(0, 100000, (num_sample_per_prompt,))
    sample_seeds = sorted(sample_seeds.numpy().tolist())
    
    generator = []
    for seed in sample_seeds:
        generator_temp = torch.Generator(device=accelerator.device)
        generator_temp.manual_seed(seed)
        generator.append(generator_temp)
    with torch.no_grad():
        output = pipeline(
            stage = stage,
            prompt = prompt,
            image_prompt = ref_images,
            prev_prompt = ref_prompt,
            height = 512,
            width = 512,
            generator = generator,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            image_guidance_scale = image_guidance_scale,
            num_images_per_prompt=num_sample_per_prompt,
        ).images
    
    images = []
    for i, image in enumerate(output):
        images.append(image[0])
        image[i].save(os.path.join(logdir, f"{frame_number}_output.png"))




import os

def iterate_frames(category_dir: str, image_dir: str, output_dir: str):
    categories = ["African", "Bloom", "Book", "Digital", "Literacy", "StoryWeaver"]
    prev_p = []  # Initialize an empty list for previous prompts within the same story
    
    for category in categories:
        category_path = os.path.join(category_dir, category)
        if not os.path.exists(category_path):
            continue
        
        for story_folder in os.listdir(category_path):
            story_path = os.path.join(category_path, story_folder)
            if not os.path.isdir(story_path):
                continue

            story_output_dir = os.path.join(output_dir, category, story_folder)
            if not os.path.exists(story_output_dir):
                os.makedirs(story_output_dir)
            
            prev_p.clear()  # Clear previous prompts for each new story
            
            # Iterate through frames in the story folder
            for frame_file in sorted(os.listdir(story_path)):
                if not frame_file.endswith(".txt"):
                    continue
                
                frame_number = frame_file.split("_")[1].split(".")[0]

                if frame_number == "0001": #first frame of every story
                    current_stage = "no"  
                    prev_p = [""]
                else:
                    current_stage = "auto-regressive"

                with open(os.path.join(story_path, frame_file), 'r') as f:
                    prompt = f.read().strip()
                
                # Prepare ref_image path based on story and frame number
                ref_image_path = os.path.join(image_dir, category, story_folder, f"{story_folder}_{frame_number}.jpg")
                
                # Call test function with current prompt and ref_image
                test(pretrained_model_path,
                     story_output_dir,
                     prompt,
                     prev_p,
                     [ref_image_path],
                     num_inference_steps,
                     guidance_scale,
                     image_guidance_scale,
                     num_sample_per_prompt,
                     stage,
                     mixed_precision,
                     frame_number=f"{story_folder}_{frame_number}")  # Pass frame_number for saving purposes
                
                # Append current prompt to prev_p list for next frame in the same story
                prev_p.append(prompt)

if __name__ == "__main__":
    pretrained_model_path = '/content/StoryGen/checkpoints/checkpoint_StorySalon'
    output_dir = "/content/output_StorySalon_Ebooks"
    num_inference_steps = 40
    guidance_scale = 7
    image_guidance_scale = 3.5
    num_sample_per_prompt = 1
    mixed_precision = "fp16"
    stage = 'auto-regressive'  # ["multi-image-condition", "auto-regressive", "no"]
    
    category_dir = "/content/StorySalon/Text/Caption"
    image_dir = "/content/StorySalon/Image_inpainted"
    
    iterate_frames(category_dir, image_dir, output_dir)

# CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py
