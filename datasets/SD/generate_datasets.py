# DUO
# Cpyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import os
from PIL import Image
from tqdm import tqdm

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler, DDIMInverseScheduler

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument("--save_dir", type=str, default="")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

device = args.device
weight_dtype = torch.float16
save_dir = args.save_dir

base_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=weight_dtype,
).to(device)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=weight_dtype,
).to(device)

base_pipe.safety_checker = None
pipe.safety_checker = None

exp_dict = {
    "Nudity": {
        "prompt": ["a naked white woman, realistic photo" for _ in range(64 // 2)]
        + ["a naked black woman, realistic photo" for _ in range(64 // 2)],
        "base_prompt": [
            "a dressed white woman, realistic photo" for _ in range(64 // 2)
        ]
        + ["a dressed black woman, realistic photo" for _ in range(64 // 2)],
        "images": "unsafe",
        "base_images": "safe",
    },
    "Blood": {
        "prompt": ["a white man, realistic photo, bleeding" for _ in range(32 // 2)]
        + ["a black man, realistic photo, bleeding" for _ in range(32 // 2)],
        "base_prompt": [
            "a white man, realistic photo, in comfort" for _ in range(32 // 2)
        ]
        + ["a black man, realistic photo, in comfort" for _ in range(32 // 2)],
        "images": "unsafe",
        "base_images": "safe",
    },
    "Suffer": {
        "prompt": ["a white man, realistic photo, suffering" for _ in range(32 // 2)]
        + ["a black man, realistic photo, suffering" for _ in range(32 // 2)],
        "base_prompt": [
            "a white man, realistic photo, enjoying" for _ in range(32 // 2)
        ]
        + ["a black man, realistic photo, enjoying" for _ in range(32 // 2)],
        "images": "unsafe",
        "base_images": "safe",
    },
    "Gun": {
        "prompt": ["a white man, realistic photo, with a gun" for _ in range(32 // 2)]
        + ["a black man, realistic photo, with a gun" for _ in range(32 // 2)],
        "base_prompt": [
            "a white man, realistic photo, with a peace" for _ in range(32 // 2)
        ]
        + ["a black man, realistic photo, with a peace" for _ in range(32 // 2)],
        "images": "unsafe",
        "base_images": "safe",
    },
    "Horror": {
        "prompt": "horror",
        "base_prompt": "comfort",
        "images": "unsafe",
        "base_images": "safe",
    },
}

import json

with open("config.json", "w") as fp:
    json.dump(exp_dict, fp, indent=2)

##########
# nudity #
##########
nudenet_labels = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]

import numpy as np
from tqdm import tqdm
from PIL import Image
from nudenet import NudeDetector

nude_detector = NudeDetector()


def is_nsfw(img):
    nude_dict = nude_detector.detect(np.array(img))
    for item in nude_dict:
        if item["class"] in nudenet_labels:
            return True
    return False


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
    base_pipe.set_progress_bar_config(disable=True)
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
        while detect_nsfw:
            edit_image = pipe(
                prompt=base_prompt[i],
                negative_prompt=prompt[i],
                image=image,
                strength=0.75,
                guidance_scale=7.5,
            )[0][0]
            detect_nsfw = is_nsfw(edit_image)

        edit_image.save(os.path.join(safe_image_dir, "{0:02d}.jpg".format(i)))

############
# violence #
############
for concept, prompt_dict in exp_dict.items():

    if concept not in ["Blood", "Suffer", "Gun", "Horror"]:
        continue

    prompt = prompt_dict["prompt"]
    base_prompt = prompt_dict["base_prompt"]
    num_images = len(prompt)

    unsafe_image_dir = os.path.join(save_dir, f"{concept}/unsafe")
    safe_image_dir = os.path.join(save_dir, f"{concept}/safe")

    os.makedirs(unsafe_image_dir, exist_ok=True)
    os.makedirs(safe_image_dir, exist_ok=True)

    # image: remove concept
    base_pipe.set_progress_bar_config(disable=True)
    for i in tqdm(range(num_images)):
        if os.path.exists(os.path.join(unsafe_image_dir, "{0:02d}.jpg".format(i))):
            continue

        image = base_pipe(
            prompt[i] if isinstance(prompt, list) else prompt, num_images_per_prompt=1
        )[0][0]
        image.save(os.path.join(unsafe_image_dir, "{0:02d}.jpg".format(i)))

    # safety editing
    pipe.set_progress_bar_config(disable=True)
    for i in tqdm(range(num_images), desc="generating safe image..."):
        image = Image.open(os.path.join(unsafe_image_dir, "{0:02d}.jpg".format(i)))
        if os.path.exists(os.path.join(safe_image_dir, "{0:02d}.jpg".format(i))):
            continue

        edit_image = pipe(
            prompt=base_prompt[i] if isinstance(base_prompt, list) else base_prompt,
            negative_prompt=prompt[i] if isinstance(prompt, list) else prompt,
            image=image,
            strength=0.8,
            guidance_scale=7.5,
        )[0][0]
        edit_image.save(os.path.join(safe_image_dir, "{0:02d}.jpg".format(i)))
