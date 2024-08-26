#!/bin/sh
declare -a list=("chair")
root="/scratch/Quadfield/"

PYTHON_PATH="python"
model_name=field
grad_thres=0.01
omega=100
vx=150
density_thres=10.0
combine=True

cd ${root}
for scene in ${list[@]};
do
  ${PYTHON_PATH} examples/marching_cubes.py ${root}/results/${scene}/${model_name}/ 100.0 True ${omega} 0 0 ${combine} ${grad_thres} ${density_thres}
  ${PYTHON_PATH} examples/downsample_mesh.py ${root}/results/${scene}/${model_name}/mesh.ply ${vx}
done
