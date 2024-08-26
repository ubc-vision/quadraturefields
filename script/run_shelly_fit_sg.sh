#!/bin/sh
declare -a list=("khady")

PYTHON_PATH="python"
root=/scratch/Quadfield/
data_root=/scratch/nerfacc/nerf_synthetic/

max_hits=25
voxel_size=300
up_sample=2.0
scaling=0.04
num_lobes=3
num_layers=2
c_lambda=1e-5
o_lambda=1e-3
reg_type="none"
agg=0.0
omega=100
max_iterations=20000
d_thres=0.01
log2_hashmap_size=21
scale=1.5 # 2.0 for woolly and horse
batch_size=18

radiance_field="finetune"
mesh_name="mesh.ply"
exp_name=finetune_sg

cd ${root}

for scene in ${list[@]}; do
$PYTHON_PATH examples/train_fit_sg.py \
--scene ${scene} \
--data_root ${data_root} \
--root ${root} \
--exp_name ${exp_name} \
--scaling ${scaling} \
--mesh_path results/${scene}/${radiance_field}/${mesh_name} \
--up_sample ${up_sample} \
--optix 0 \
--voxel_size ${voxel_size} \
--max_hits ${max_hits} \
--num_lobes ${num_lobes} \
--num_layers ${num_layers} \
--ckpt_path ckpts/${scene}/${radiance_field}/model.pth \
--reg_type ${reg_type} \
--c_lambda ${c_lambda} \
--o_lambda ${o_lambda} \
--agg ${agg} \
--max_iterations ${max_iterations} \
--log2_hashmap_size ${log2_hashmap_size} \
--batch_size ${batch_size} \
--scale ${scale} \

done