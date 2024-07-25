import os
import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint
from torchvision import transforms
from PIL import Image
import argparse
import shutil

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StableDiffusionPipeline

from transformers import AutoProcessor, AutoModel

from transformers import LlavaNextForConditionalGeneration, LlamaTokenizer
from get_llava_important_words import get_llava_important_words

logger = get_logger(__name__)

from accelerate.utils import write_basic_config

write_basic_config()

print("0")

def calc_probs(processor, model, prompt, images):
    
    # preprocess
    image_inputs = processor(images=images, padding=True, truncation=True, max_length=77, return_tensors="pt").to('cuda')
    text_inputs = processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to('cuda')

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist()

def test(
        testdir, # PATH/TO/StorySalon_TestSet_samples/PDF/Text
        logdir = '/content/drive/My Drive/inference_Storygen', 
        task = 'continuation', # visualization or continuation
        do_llava_filter = False, 
        llava_filter_num = 7, 
):
    
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to("cuda")
    
    pretrained_model_path = '/content/StoryGen/checkpoints/checkpoint_StorySalon'
    num_inference_steps = 40
    guidance_scale = 7.0
    image_guidance_scale = 3.5
    num_sample_per_prompt = 10
    stage = "auto-regression"
    mixed_precision = "fp16"
        
    accelerator = Accelerator(mixed_precision=mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    print("1")

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )

    print("2")
    
    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed" f" correctly and a GPU is available: {e}"
            )
    unet, pipeline = accelerator.prepare(unet, pipeline)
    pipeline.set_progress_bar_config(disable=True)

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

    if do_llava_filter:
        llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf").to('cpu', dtype=weight_dtype)
        llava_model.eval()
        llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        sp_model = LlamaTokenizer.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf").sp_model
    
    print("3")

    for testID in tqdm(os.listdir(testdir)):
        story_dir = os.path.join(testdir, testID)
        ref_prompts = []
        ref_image = []

        if not os.path.isdir(story_dir):
            raise AttributeError(f"{story_dir} is not a directory. ")
        
        sub_set = "PDF" if "PDF" in testdir else "Video"
        image_path = os.path.join(logdir, sub_set, testID)

        if os.path.exists(image_path):
            task_diff = 1 if task=='continuation' else 0
            if len(os.listdir(story_dir))==(len(os.listdir(image_path))+task_diff):
                print(f"Skip for {sub_set}/{testID}.")
                continue
            shutil.rmtree(image_path)
            print(f"Redo for {sub_set}/{testID}.")
        os.makedirs(image_path)

        for idx, frame in enumerate(sorted(os.listdir(story_dir))):
            if not frame.endswith(".txt"):
                raise AttributeError(f"{frame} is not a txt file.")
            
            with open(os.path.join(story_dir, frame)) as f:
                prompt = f.read().strip()

            if idx==0 and task=='continuation':
                image_dir = story_dir.replace('Text', 'Image') 
                ref_image.append(Image.open(os.path.join(image_dir, sorted(os.listdir(image_dir))[0])))
                if do_llava_filter:
                    vae.to('cuda')
                    text_encoder.to('cuda')
                    unet.to('cuda')
                    llava_model.to(accelerator.device)
                    prompt = get_llava_important_words(prompt, Image.open(os.path.join(image_dir, sorted(os.listdir(image_dir))[0])), llava_filter_num, llava_model, llava_processor, sp_model)
                    llava_model.to('cuda')
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    text_encoder.to(accelerator.device)
                    unet.to(accelerator.device)
                    
                ref_prompts.append(prompt)
                continue
            elif idx==0:
                raise NotImplementedError
            
            ref_images= []
            for r_image in ref_image:
                r_image = r_image.convert('RGB').resize((512, 512))
                r_image = transforms.ToTensor()(r_image)
                ref_images.append(np.ascontiguousarray(r_image))
            ref_images = torch.from_numpy(np.ascontiguousarray(ref_images)).float()
            for ref_img in ref_images:
                ref_img = ref_img * 2. - 1.
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
                    prev_prompt = ref_prompts,
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
            scores = calc_probs(processor, model, prompt, images)
            index = np.argmax(scores)
            images[index].save(os.path.join(image_path, frame[:-4]+'.png'))
            
            if idx == len(os.listdir(story_dir))-1:
                break

            if do_llava_filter:
                vae.to('cuda')
                text_encoder.to('cuda')
                unet.to('cuda')
                llava_model.to(accelerator.device)
                prompt = get_llava_important_words(prompt, images[index], llava_filter_num, llava_model, llava_processor, sp_model)
                llava_model.to('cuda')
                torch.cuda.empty_cache()
                vae.to(accelerator.device)
                text_encoder.to(accelerator.device)
                unet.to(accelerator.device)

            ref_prompts.append(prompt)
            ref_image.append(images[index])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the test function with specified directories and task.')
    
    parser.add_argument('--testdir', type=str, default='/content/TestSet/Video/Text',
                        help='Path to the test directory. Default is /content/drive/MyDrive/dataset/StorySalon_TestSet_samples/PDF/Text')
    parser.add_argument('--logdir', type=str, default='/content/drive/My Drive/inference_Storygen',
                        help='Path to the log directory. Default is /content/StoryGen/inference_StoryGen')
    parser.add_argument('--task', type=str, choices=['visualization', 'continuation'], default='continuation',
                        help='Task to perform: "visualization" or "continuation". Default is "continuation"')
    parser.add_argument('--do_llava_filter', action='store_true',
                        help='Apply LLAVA filter if specified.')
    parser.add_argument('--llava_filter_num', type=int, default=7, 
                        help='LLAVA filter number. Only needed if do_llava_filter is specified.')

    args = parser.parse_args()

    test(
        testdir=args.testdir, 
        logdir=args.logdir, 
        task=args.task,
        do_llava_filter=args.do_llava_filter, 
        llava_filter_num=args.llava_filter_num, 
    )


# CUDA_VISIBLE_DEVICES=0 accelerate launch inference_storysalon.py
