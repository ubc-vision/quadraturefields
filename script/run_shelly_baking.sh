#!/bin/sh
declare -a scenes=(khady)

model_name_sg="finetune_sg"
model_name="finetune"

root=/scratch/Quadfield/
data_root=/scratch/nerfacc/nerf_synthetic/
PYTHON_PATH="python"
SEGMENTOR_PATH="/home/gopalshr/cape_coalwood/ScanNet/Segmentator/segmentator"
# scene parameters
scale=1.5 # 2.0 for woolly and horse

# model parameters
max_hits=25
up_sample=2
num_lobes=3
num_layers=2
log2_hashmap_size=21

# baking parameters
compression_type="linear"
lambda_thres=5.0
discretize=False
texture_size=8192
mesh_name=mesh_updated.ply

cd ${root}

for scene in "${scenes[@]}";
do
${PYTHON_PATH} examples/prune_mesh_after_finetuning.py --scene ${scene} --data_root ${data_root} --exp_name temp --num_lobes 0 --num_layers ${num_layers} --o_lambda 0 --ckpt ${root}/ckpts/${scene}/${model_name}/model.pth --root ${root} --mesh_path ${root}/results/${scene}/${model_name}/mesh.ply  --max_hits ${max_hits} --up_sample ${up_sample} --log2_hashmap_size ${log2_hashmap_size} --scale ${scale}
${SEGMENTOR_PATH} ${root}/results/${scene}/${model_name}/mesh_updated.ply

${PYTHON_PATH} examples/generate_uv_xatlas_old.py ${root}/results/${scene}/${model_name}/ mesh_updated.ply mesh_updated.0.010000.segs.json ${texture_size} False

${PYTHON_PATH} examples/bake_texture_images_shelly.py --root ${root} --data_root ${data_root} --scene ${scene} --ckpt_path ${root}/ckpts/${scene}/${model_name}/model.pth --ckpt_path_sg ${root}/ckpts/${scene}/${model_name_sg}/model.pth --num_lobes ${num_lobes} --num_layers ${num_layers} --mesh_path ${root}/results/${scene}/${model_name}/"${mesh_name%.*}"/mesh_segmentation_${texture_size}.obj --optix False --texture_size ${texture_size} --log2_hashmap_size ${log2_hashmap_size} --scale ${scale} --compression_type ${compression_type} --lambda_thres ${lambda_thres}

${PYTHON_PATH} examples/test_baking_texture_images.py --root ${root} --data_root ${data_root} --scene ${scene} --ckpt_path ${root}/ckpts/${scene}/${model_name}/model.pth --num_lobes ${num_lobes} --num_layers ${num_layers} --mesh_path ${root}/results/${scene}/${model_name}/"${mesh_name%.*}"/mesh_segmentation_${texture_size}.obj --max_hits ${max_hits} --up_sample ${up_sample} --optix False --discretize False --texture_size ${texture_size} --log2_hashmap_size ${log2_hashmap_size}  --scale ${scale}  --compression_type ${compression_type} --lambda_thres ${lambda_thres}

done
