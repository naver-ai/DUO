# DUO
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import huggingface_hub
import torch
import os
import argparse
from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument("--save_dir", type=str, default="")
parser.add_argument("--device0", type=str, default="cuda:0")
parser.add_argument("--device1", type=str, default="cuda:1")
parser.add_argument("--token", type=str, default="")
args = parser.parse_args()

if args.token != "":
    huggingface_hub.login(args.token)

device0 = "cuda:0"
device1 = "cuda:1"
weight_dtype = torch.float16
save_dir = args.save_dir

base_pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=weight_dtype
).to(device0)
pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=weight_dtype
).to(device1)

exp_dict = {
    "Nudity": {
        "prompt": ["a naked white woman, realistic photo" for _ in range(64 // 2)]
        + ["a naked black woman, realistic photo" for _ in range(64 // 2)],
        "base_prompt": ["a white woman, realistic photo" for _ in range(64 // 2)]
        + ["a black woman, realistic photo" for _ in range(64 // 2)],
        "images": "unsafe",
        "base_images": "safe",
    },
}

import json

with open("config.json", "w") as fp:
    json.dump(exp_dict, fp, indent=2)

##########
# Nudity #
##########
all_labels = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]

import numpy as np
from PIL import Image
from nudenet import NudeDetector

nude_detector = NudeDetector()


def is_nsfw(edit_image):
    nude_dict = nude_detector.detect(np.array(edit_image))
    nude_item = 0
    for item in nude_dict:
        if item["class"] in all_labels:
            nude_item += 1
    if nude_item == 0:
        return False
    else:
        return True


base_pipe.safety_checker = None
pipe.safety_checker = None

for concept, prompt_dict in exp_dict.items():

    if concept != "Nudity":
        continue

    prompt = prompt_dict["prompt"]
    base_prompt = prompt_dict["base_prompt"]
    num_images = len(prompt)

    unsafe_image_dir = os.path.join(save_dir, f"{concept}/unsafe")
    safe_image_dir = os.path.join(save_dir, f"{concept}/safe")

    os.makedirs(unsafe_image_dir, exist_ok=True)
    os.makedirs(safe_image_dir, exist_ok=True)

    # image: remove concept
    for i in tqdm(range(num_images)):
        if os.path.exists(os.path.join(unsafe_image_dir, "{0:02d}.jpg".format(i))):
            continue

        detect_nsfw = False
        while not detect_nsfw:
            image = base_pipe(prompt[i], num_images_per_prompt=1)[0][0]
            detect_nsfw = is_nsfw(image)
        image.save(os.path.join(unsafe_image_dir, "{0:02d}.jpg".format(i)))

    # safety editing
    pipe.set_progress_bar_config(disable=True)
    for i in tqdm(range(num_images), desc="generating safe image..."):
        image = Image.open(os.path.join(unsafe_image_dir, "{0:02d}.jpg".format(i)))
        if os.path.exists(os.path.join(safe_image_dir, "{0:02d}.jpg".format(i))):
            continue

        detect_nsfw = True
        num_try = 0
        while detect_nsfw:
            edit_image = pipe(
                prompt=base_prompt[i],
                negative_prompt=prompt[i],
                image=image,
                strength=0.75,
                guidance_scale=7.5,
            )[0][0]
            detect_nsfw = is_nsfw(image)
        edit_image.save(os.path.join(safe_image_dir, "{0:02d}.jpg".format(i)))
