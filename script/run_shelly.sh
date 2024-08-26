#!/bin/sh

PYTHON_PATH="python"
data_root="/scratch/nerfacc/nerf_synthetic/"
root="/scratch/Quadfield/"
declare -a list=("khady")
exp_name="nerf"
num_lobes=0
o_lambda=0.0001
occ_thres=0.01
num_layers=2
scale=1.5 # 2.0 for woolly and horse
reg_type="entropy"
log2_hashmap_size=21
batch_size=22
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
