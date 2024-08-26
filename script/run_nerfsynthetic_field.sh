#!/bin/sh
PYTHON_PATH="python"
data_root="/scratch/nerfacc/nerf_synthetic/"
root="/scratch/Quadfield/"

declare -a list=(chair)
exp_name="field"
num_lobes=0
# increase the batch size for better results at the cost of memory
batch_size=19
log2_hashmap_size=19
occ_thres=0.005
scale=1.5
max_steps=25000
cd $root

for scene in ${list[@]};
do
modelname="nerf"
${PYTHON_PATH} examples/train_field.py \
 --scene ${scene} \
 --data_root ${data_root} \
 --root ${root} \
 --ckpt_path ${root}/ckpts/${scene}/${modelname}/ngp.pth \
 --exp_name ${exp_name} \
 --num_lobes ${num_lobes} \
 --num_layers 2  \
 --max_steps ${max_steps} \
 --log2_hashmap_size ${log2_hashmap_size} \
 --batch_size ${batch_size} \
 --occ_thres ${occ_thres} \
 --scale ${scale}

done
