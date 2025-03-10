export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
port=50000

config_name="Nudity"
num_samples=64
method='DPO'
exp_name='nudity'

base_dir=$(pwd)
save_dir="./outputs"
ckpt=1000

# for dcoloss_beta in 2000 1000 500 250 100 ; do
for dcoloss_beta in 500 ; do

cd $base_dir/train
lora_dir="$save_dir/unlearn/SD3-train/dpo/$dcoloss_beta/$config_name"

accelerate launch                                       \
    --main_process_port $port                           \
    unlearn-sd3.py                                      \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --project="SD3-DPO_survival-no_prompt"              \
    --group=""                                          \
    --config_dir="$base_dir/datasets/SD3/config.json"   \
    --config_name="${config_name}"                      \
    --data_dir="$base_dir/datasets/SD3"                 \
    --output_dir="${lora_dir}"                          \
    --target_prompt="${target_prompt}"                  \
    --synonym_prompt="${synonym_prompt}"                \
    --prior_prompt="${prior_prompt}"                    \
    --base_lr=3e-5                                      \
    --adam_weight_decay=1e-2                            \
    --dcoloss_beta=$dcoloss_beta                        \
    --base_lambda=1e6                                   \
    --rank=32                                           \
    --method=dpo                                        \
    --train_batch_size=1                                \
    --max_train_steps=1000                              \
    --checkpointing_steps=1000                          \
    --validation_steps=1000                             \
    --num_validation_images=2                           \
    --num_samples=$num_samples                          \
    --t_max=750                                         \
    --t_min=1                                           \
    --mixed_precision="fp16"                            \
    --seed=42

done
