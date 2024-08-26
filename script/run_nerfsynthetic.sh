#!/bin/sh

PYTHON_PATH="python"
data_root="/scratch/nerfacc/nerf_synthetic/"
root="/scratch/Quadfield/"
declare -a list=("chair")
exp_name="nerf"
num_lobes=0
o_lambda=0.001
occ_thres=0.01
num_layers=2
scale=1.5
reg_type="occ"
log2_hashmap_size=19
# increase the batch size for better results at the cost of memory
batch_size=20
max_steps=20000

cd $root

for scene in ${list[@]};
do
 $PYTHON_PATH examples/train_ngp_nerf_sg_occ.py \
 --root ${root} \
 --scene ${scene} \
 --data_root ${data_root} \
 --exp_name ${exp_name} \
 --num_lobes ${num_lobes} \
 --num_layers ${num_layers} \
 --o_lambda ${o_lambda} \
 --occ_thres ${occ_thres} \
 --log2_hashmap_size ${log2_hashmap_size} \
 --batch_size ${batch_size} \
 --scale ${scale} \
 --reg_type ${reg_type} \
 --max_steps ${max_steps}

done
