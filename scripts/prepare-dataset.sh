export PATH="$HOME/.local/bin:$PATH"

base_dir=$(pwd)

cd $base_dir/datasets/SD
python3 generate_datasets.py \
    --save_dir  $base_dir/datasets/SD   \
    --device    "cuda:0"                \

cd $base_dir/datasets/SD3
python3 generate_datasets_sd3.py \
    --save_dir  $base_dir/datasets/SD3  \
    --device0   "cuda:0"                \
    --device1   "cuda:1"                \
