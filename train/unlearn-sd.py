# DUO
# Copyright (c) 202-present NAVER Cloud Corp.
# Apache-2.04
#
# Heavily adopted code from DCO
# ref: https://github.com/kyungmnlee/dco

import argparse
import itertools
import logging
import math
import os
import shutil
import random
from pathlib import Path
import json
import torch
import torch.nn.functional as F

import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    is_wandb_available,
)

print("finish import libs")

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # pretrained model config
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument(
        "--no_cross_attn",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    # data config
    parser.add_argument(
        "--dynamic_lambda",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--given_prompt",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--cfg_train",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument("--no_grad", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--config_dir", type=str, default="")
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--project", type=str, default="unlearning")
    parser.add_argument("--num_samples", type=int, default=None)
    # validation config
    parser.add_argument(
        "--prior_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is close to prior distribution.",
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--synonym_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=5,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument("--validation_steps", type=int, default=500)
    # use prior preservation
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    # save config
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outdir",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    # dataloader config
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this",
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    # train config
    parser.add_argument(
        "--method", type=str, default="", help="if kto, detach prior loss."
    )
    parser.add_argument(
        "--t_max", type=int, default=1000, help="if kto, detach prior loss."
    )
    parser.add_argument(
        "--t_min", type=int, default=0, help="if kto, detach prior loss."
    )
    parser.add_argument(
        "--base_lambda",
        type=float,
        default=7.5,
        help="Additional prior preservation regularization loss",
    )
    # parser.add_argument("--ourloss_lambda", type=float, default=7.5, help="Additional prior preservation regularization loss")
    parser.add_argument("--ddsloss_eta", type=float, default=7.5, help="")
    parser.add_argument(
        "--dcoloss_beta",
        type=float,
        default=1000,
        help="Sigloss value for DCO loss, use -1 if do not using dco loss",
    )

    parser.add_argument(
        "--train_text_encoder_ti",
        action="store_true",
        help=("Whether to use textual inversion"),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    # optimizer config
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    # optimizer config
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=None,
        help="Weight decay to use for text_encoder",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    # save config
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--offset_noise", type=float, default=0.0)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class TrainDataset(Dataset):
    def __init__(self, args):
        self.method = args.method
        self.size = args.resolution
        self.center_crop = args.center_crop
        self.config_dir = args.config_dir
        self.data_dir = args.data_dir
        self.config_name = args.config_name
        self.train_with_dco_loss = args.dcoloss_beta > 0.0
        self.with_prior_preservation = args.with_prior_preservation

        with open(self.config_dir, "r") as data_config:
            data_cfg = json.load(data_config)[self.config_name]

        instance_image_dir = os.path.join(
            self.data_dir, self.config_name, data_cfg["images"]
        )
        base_image_dir = os.path.join(
            self.data_dir, self.config_name, data_cfg["base_images"]
        )
        instance_image_path_list = os.listdir(instance_image_dir)
        base_image_path_list = os.listdir(base_image_dir)
        instance_image_path_list.sort(reverse=False)
        base_image_path_list.sort(reverse=False)

        instance_image_path_list = instance_image_path_list[: args.num_samples]
        base_image_path_list = base_image_path_list[: args.num_samples]
        print(f"instance_image_path_list: {instance_image_path_list}")
        print(f"base_image_path_list: {base_image_path_list}")

        self.instance_images = [
            Image.open(os.path.join(instance_image_dir, path))
            for path in instance_image_path_list
        ]
        self.base_images = [
            Image.open(os.path.join(base_image_dir, path))
            for path in base_image_path_list
        ]

        if self.method == "kto":
            random.shuffle(self.base_images)

        if args.given_prompt:
            self.instance_prompts = [
                data_cfg["prompt"][i]
                if isinstance(data_cfg["prompt"], list)
                else data_cfg["prompt"]
                for i, _ in enumerate(instance_image_path_list)
            ]
            self.base_prompts = [
                data_cfg["base_prompt"][i]
                if isinstance(data_cfg["base_prompt"], list)
                else data_cfg["base_prompt"]
                for i, _ in enumerate(base_image_path_list)
            ]
        else:
            self.instance_prompts = ["" for i in instance_image_path_list]
            self.base_prompts = ["" for i in base_image_path_list]

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(self.size)
                if self.center_crop
                else transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        # instance image (image for concept removal)
        instance_image = self.instance_images[index % self.num_instance_images]
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        # base image (image for prior preservation)
        base_image = self.base_images[index % self.num_instance_images]
        base_image = exif_transpose(base_image)

        if not base_image.mode == "RGB":
            base_image = base_image.convert("RGB")
        example["base_images"] = self.image_transforms(base_image)

        # instance prompt
        prompt = self.instance_prompts[index % self.num_instance_images]
        example["instance_prompt"] = prompt

        # base prompt
        base_prompt = self.base_prompts[index % self.num_instance_images]
        example["base_prompt"] = base_prompt

        return example


def collate_fn(examples, args):
    pixel_values = [example["instance_images"] for example in examples]
    base_pixel_values = [example["base_images"] for example in examples]

    prompts = [example["instance_prompt"] for example in examples]
    base_prompts = [example["base_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    base_pixel_values = torch.stack(base_pixel_values)
    base_pixel_values = base_pixel_values.to(
        memory_format=torch.contiguous_format
    ).float()

    return {
        "pixel_values": pixel_values,
        "base_pixel_values": base_pixel_values,
        "prompts": prompts,
        "base_prompts": base_prompts,
    }


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    ###############
    # preparation #
    ###############
    # logging
    logging_dir = Path(args.output_dir, args.logging_dir)

    # accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

        wandb.login(key="enter_your_wandb_key")
        if args.dcoloss_beta > 0:
            exp_name = f"SD-{args.config_name}-{args.method}-beta_{args.dcoloss_beta}-lambda_{args.base_lambda}-lr_{args.base_lr}-lora_rank_{args.rank}-bs_{args.train_batch_size}-num_samples_{args.num_samples}"
            if not args.given_prompt:
                exp_name += "-no_prompt"
            if args.cfg_train:
                exp_name += "-cfg_train"
            if args.no_cross_attn:
                exp_name += "-no_cross_attn"
            if args.t_max != 1000:
                exp_name += f"-t_max_{args.t_max}"
            if args.t_min != 1:
                exp_name += f"-t_min_{args.t_min}"
            if args.dynamic_lambda:
                exp_name += "-dynamic_lambda"
            if args.no_grad != "":
                exp_name += f"-no_grad_{args.no_grad}"
        elif args.dcoloss_beta == 0:
            exp_name = f"{args.config_name}-naive_unlearning"

        if accelerator.is_main_process:
            wandb.init(
                project=args.project,
                name=exp_name,
                # Track hyperparameters and run metadata
                config={
                    "group": args.group,
                    "config_name": args.config_name,
                    "method": args.method,
                    "base_lambda": args.base_lambda,
                    "dcoloss_beta": args.dcoloss_beta,
                },
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    ##############
    # Load Model #
    ##############
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=weight_dtype
    )
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    text_encoder_one = pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet = pipe.unet.to(accelerator.device, dtype=weight_dtype)
    vae = pipe.vae.to(
        accelerator.device, dtype=torch.float32
    )  # to avoid precision error

    # We only train the additional adapter LoRA layers
    text_encoder_one.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    #############
    # Load LoRA #
    #############
    # now we will add new LoRA weights to the attention layers
    if args.no_cross_attn:
        target_modules = ["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0"]
    else:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        assert False
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.append(text_encoder_one)
        for model in models:
            for param in model.parameters():
                # only upcast trainable parameters (LoRA) into fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(
                    model, type(accelerator.unwrap_model(text_encoder_one))
                ):
                    if args.train_text_encoder:
                        text_encoder_one_lora_layers_to_save = (
                            convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(model)
                            )
                        )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(
            lora_state_dict, network_alphas=network_alphas, unet=unet_
        )

        text_encoder_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder." in k
        }
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder_one_,
        )

    accelerator.register_save_state_pre_hook(
        save_model_hook
    )  # save_model_hook is called in Accelerator.save_state() before save_checkpoint.
    accelerator.register_load_state_pre_hook(
        load_model_hook
    )  # return handles to handle.remove()

    ###########
    # dataset #
    ###########
    # Dataset and DataLoaders creation:
    train_dataset = TrainDataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args),
        num_workers=args.dataloader_num_workers,
    )

    #############
    # optimizer #
    #############
    # rescale learning rate
    args.learning_rate = args.base_lr * (100 / args.dcoloss_beta)
    args.ourloss_lambda = args.base_lambda * (args.dcoloss_beta / 100)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))

    if args.train_text_encoder:
        text_lora_parameters_one = list(
            filter(lambda p: p.requires_grad, text_encoder_one.parameters())
        )

    # If neither --train_text_encoder nor --train_text_encoder_ti, text_encoders remain frozen during training
    freeze_text_encoder = not (args.train_text_encoder or args.train_text_encoder_ti)

    # Optimization parameters
    unet_lora_parameters_with_lr = {
        "params": unet_lora_parameters,
        "lr": args.learning_rate,
    }

    if not freeze_text_encoder:
        # different learning rate for text encoder and unet
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder
            if args.adam_weight_decay_text_encoder
            else args.adam_weight_decay,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            unet_lora_parameters_with_lr,
            text_lora_parameters_one_with_lr,
        ]
    else:
        params_to_optimize = [unet_lora_parameters_with_lr]

    # Optimizer creation
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if not freeze_text_encoder:
        (
            unet,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("fine-tune sdxl", config=vars(args))

    ##############
    # validation #
    ##############
    def validation(args, unet, prompts, titles):
        # create pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            # vae=vae,
            # text_encoder=accelerator.unwrap_model(text_encoder_one),
            # text_encoder_2=accelerator.unwrap_model(text_encoder_two),
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline.safety_checker = None

        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config, **scheduler_args
        )

        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference (validation prompt)
        for prompt, title in zip(prompts, titles):
            if prompt != "":
                generator = (
                    torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    if args.seed
                    else None
                )
                pipeline_args = {"prompt": prompt}

                with torch.cuda.amp.autocast():
                    images = [
                        pipeline(**pipeline_args, generator=generator).images[0]
                        for _ in range(args.num_validation_images)
                    ]

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                f"{title}": [
                                    wandb.Image(image, caption=f"{i}: {prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

        del pipeline
        torch.cuda.empty_cache()

    ###############
    # !!!Train!!! #
    ###############
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    validation(
        args,
        unet,
        prompts=[
            args.target_prompt,
            args.synonym_prompt,
            args.prior_prompt,
        ],
        titles=[
            "validation-lora_on",
            "synonym-lora_on",
            "prior-lora_on",
        ],
    )
    # for epoch in range(first_epoch, args.num_train_epochs):
    for epoch in range(first_epoch, args.num_train_epochs):
        # if performing any kind of optimization of text_encoder params
        if args.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            if args.train_text_encoder:
                text_encoder_one.text_model.embeddings.requires_grad_(True)

        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                prompts = batch["prompts"]
                base_prompts = batch["base_prompts"]
                null_prompts = ["" for _ in prompts]

                if args.cfg_train:
                    for i in range(len(prompts)):
                        if random.random() > 0.8:
                            prompts[i] = ""
                            base_prompts[i] = ""

                # encode batch prompts when custom prompts are provided for each image -
                # if train_dataset.custom_instance_prompts:
                if freeze_text_encoder:
                    prompt_embeds, _ = pipe.encode_prompt(
                        prompts + base_prompts + null_prompts,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                else:
                    assert False, "we freeze text encoder for unlearning"

                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                base_pixel_values = batch["base_pixel_values"].to(dtype=vae.dtype)
                pixel_values = torch.cat([pixel_values, base_pixel_values], dim=0)

                with torch.no_grad():
                    model_input = (
                        vae.encode(pixel_values).latent_dist.sample()
                        * vae.config.scaling_factor
                    )
                model_input = model_input.to(weight_dtype)

                # Sample noise that we'll add to the latents
                bsz = pixel_values.shape[0]
                noise = torch.randn_like(model_input[: bsz // 2]).repeat(2, 1, 1, 1)

                # Sample a random timestep for each image
                if args.method == "kto":
                    timesteps = torch.randint(
                        max(args.t_min - 1, 0),
                        min(args.t_max, noise_scheduler.config.num_train_timesteps),
                        (1,),
                        device=model_input.device,
                    ).repeat(bsz)
                elif args.method == "dpo":
                    timesteps = torch.randint(
                        max(args.t_min - 1, 0),
                        min(args.t_max, noise_scheduler.config.num_train_timesteps),
                        (bsz,),
                        device=model_input.device,
                    )
                prior_timesteps = torch.randint(
                    999, 1000, (bsz // 2,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                noisy_model_input = noise_scheduler.add_noise(
                    model_input, noise, timesteps
                )

                timesteps = torch.cat([timesteps, prior_timesteps], dim=0)
                noisy_model_input = torch.cat(
                    [noisy_model_input, noise[: bsz // 2]], dim=0
                )

                assert timesteps.size(0) == noisy_model_input.size(0)

                # Predict the noise residual
                if freeze_text_encoder:
                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                    ).sample

                    if args.dcoloss_beta > 0.0:
                        with torch.no_grad():
                            if torch.cuda.device_count() > 1:
                                unet.module.disable_adapters()
                            else:
                                unet.disable_adapters()
                            refer_pred = unet(
                                noisy_model_input,
                                timesteps,
                                prompt_embeds,
                            ).sample
                            if torch.cuda.device_count() > 1:
                                unet.module.enable_adapters()
                            else:
                                unet.enable_adapters()

                if base_prompts is not None:
                    model_pred, model_base, model_prior = model_pred.chunk(3)
                    refer_pred, refer_base, refer_prior = refer_pred.chunk(3)
                else:
                    raise ValueError("need base_prompts")

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                if args.snr_gamma is None:
                    if args.dcoloss_beta > 0.0:
                        loss_model_pred = F.mse_loss(
                            model_pred.float(),
                            target[: bsz // 2].float(),
                            reduction="none",
                        ).mean(dim=[1, 2, 3])
                        loss_refer_pred = F.mse_loss(
                            refer_pred.float(),
                            target[: bsz // 2].float(),
                            reduction="none",
                        ).mean(dim=[1, 2, 3])
                        loss_model_base = F.mse_loss(
                            model_base.float(),
                            target[: bsz // 2].float(),
                            reduction="none",
                        ).mean(dim=[1, 2, 3])
                        loss_refer_base = F.mse_loss(
                            refer_base.float(),
                            target[: bsz // 2].float(),
                            reduction="none",
                        ).mean(dim=[1, 2, 3])
                        loss_base = (
                            (loss_model_base - loss_refer_base).mean().unsqueeze(0)
                            if args.method == "kto"
                            else (loss_model_base - loss_refer_base)
                        )
                        loss_pred = loss_model_pred - loss_refer_pred
                        if args.no_grad == "ascent":
                            loss_pred = loss_pred.detach()
                        if args.no_grad == "descent":
                            loss_base = loss_base.detach()

                        diff = loss_base - loss_pred
                        inside_term = -1 * args.dcoloss_beta * diff
                        loss = -1 * torch.nn.LogSigmoid()(inside_term)
                        loss = loss.mean()
                    else:
                        loss = -F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )

                    if args.ourloss_lambda > 0.0:
                        if args.dynamic_lambda and loss < 0.1:
                            lambda_ = 100 * args.ourloss_lambda
                        else:
                            lambda_ = args.ourloss_lambda
                        prior_preservation = F.mse_loss(
                            model_prior.float(), refer_prior.float(), reduction="mean"
                        )
                        loss += lambda_ * prior_preservation
                    else:
                        pass

                else:
                    raise ValueError()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(
                            unet_lora_parameters,
                            text_lora_parameters_one,
                        )
                        if (args.train_text_encoder or args.train_text_encoder_ti)
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                logs = {
                    "loss": loss.mean().detach().item(),
                    "loss_base": loss_base.mean().detach().item(),  # prior
                    "loss_pred": loss_pred.mean().detach().item(),  # unlearn
                    "diff": diff.mean().detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if args.ourloss_lambda > 0.0:
                    logs["loss_prior"] = prior_preservation.detach().item()

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step % args.validation_steps == 0:
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompts"
                    )
                    if torch.cuda.device_count() > 1:
                        unet.module.enable_adapters()
                    else:
                        unet.enable_adapters()
                    validation(
                        args,
                        unet,
                        prompts=[
                            args.target_prompt,
                            args.synonym_prompt,
                            args.prior_prompt,
                        ],
                        titles=[
                            "validation-lora_on",
                            "synonym-lora_on",
                            "prior-lora_on",
                        ],
                    )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unet)
        )

        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(text_encoder_one)
            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            )
        else:
            text_encoder_lora_layers = None

        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
        )
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(
        os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
    ):
        pass
    else:
        main(args)
