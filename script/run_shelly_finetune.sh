#!/bin/sh
declare -a list=("khady")
PYTHON_PATH="python"
root=/scratch/Quadfield/
data_root=/scratch/nerfacc/nerf_synthetic/

max_hits=25
voxel_size=150
up_sample=2.0
scaling=0.04
num_lobes=0
num_layers=2
c_lambda=1e-5
o_lambda=1e-3
reg_type="none"
agg=0.0
omega=100
max_iterations=10000
d_thres=0.001
log2_hashmap_size=21
scale=1.5 # 2.0 for woolly and horse

# reduce it to save GPU memory
# larger batch size improves performance
batch_size=17

field_model_name="field"
radiance_field="nerf"
mesh_name="smp_mesh.ply"

exp_name=finetune

cd ${root}
for scene in ${list[@]}; do

${PYTHON_PATH} examples/train_finetune.py \
--scene ${scene} \
--data_root ${data_root} \
--root ${root} \
--exp_name ${exp_name} \
--scaling ${scaling} \
--mesh_path ${root}/results/${scene}/${field_model_name}/${mesh_name} \
--up_sample ${up_sample} \
--optix 0 \
--voxel_size ${voxel_size} \
--max_hits ${max_hits} \
--num_lobes ${num_lobes} \
--num_layers ${num_layers} \
--ckpt_path ${root}/ckpts/${scene}/${radiance_field}/ngp.pth \
--reg_type ${reg_type} \
--c_lambda ${c_lambda} \
--o_lambda ${o_lambda} \
--agg ${agg} \
--max_iterations ${max_iterations} \
--log2_hashmap_size ${log2_hashmap_size} \
--batch_size ${batch_size} \
--scale ${scale} \

done