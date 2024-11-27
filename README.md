# DUO: Direct Unlearning Optimization for Robust and Safe Text-to-Image Models

This repository contains the official implementation of [Direct Unlearning Optimization for Robust and Safe Text-to-Image Models](https://arxiv.org/abs/2407.21035).


**Our work is accepted at :tada:NeurIPS 2024:tada:!** This repository code is based on the settings used for the NeurIPS submission and differs from the configuration in the arXiv paper. The NeurIPS version of the paper will be uploaded to arXiv soon.

## Setup

### Prerequisites

- Docker
- NVIDIA GPU with CUDA support (4-GPU is recommended)

### Environment Setup

1. Use the following Docker container image:
   ```
   nvcr.io/nvidia/pytorch:23.12-py3
   ```

2. After setting up the Docker container, run:
   ```bash
   git clone https://github.com/naver-ai/DUO
   cd duo 
   pip install -r requirements.txt
   accelerate config
   ```

## Training

### Dataset preparation
```bash
bash scripts/prepare-dataset.sh
```

### Stable Diffusion 1.4v
#### Nudity Experiment
```bash
bash scripts/sd-nudity.sh
```

#### Violence Experiment
```bash
bash scripts/sd-violence.sh
```

- We use `train_batch_size=4`. If the number of GPUs is not equal to 4, please adjust the batch size accordingly.
- Specify the `save_dir` in all scripts. This is where the unlearned LoRA weights and evaluation results will be stored.
- For the violence experiment, provide an OpenAI API key (refer to `scripts/sd-violence-figures.sh`).

### Stable Diffusion 3

```bash
bash scripts/sd3-nudity.sh
```

## Inference
We provide simple inference codes in `inference.py`.
```bash
python3 inference.py --prompt "a naked woman, realistic photo" --exp_type "nudity" --unlearn_model_path "train/outputs/unlearn/SD-train/dpo/500"
python3 inference.py --prompt "a man bleeding" --exp_type "violence" --unlearn_model_path "train/outputs/unlearn/SD-train/dpo/1000"
```

## Acknowledgements

This repository is heavily based on the following projects:
- [Diffusers](https://github.com/huggingface/diffusers)
- [DCO](https://github.com/kyungmnlee/dco)
- [LoRA](https://github.com/cloneofsimo/lora/tree/master)

## License
```
DUO
Copyright (c) 2024-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
