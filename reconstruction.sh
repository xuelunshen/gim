#!/bin/bash
scene_name=$1
version=$2

python reconstruction.py --scene_name ${scene_name} --version ${version}

# gim
colmap image_undistorter \
    --image_path inputs/${scene_name}/images \
    --input_path outputs/${scene_name}/${version}/sparse \
    --output_path outputs/${scene_name}/${version}/dense

colmap patch_match_stereo \
    --workspace_path outputs/${scene_name}/${version}/dense

colmap stereo_fusion \
    --workspace_path outputs/${scene_name}/${version}/dense \
    --output_path outputs/${scene_name}/${version}/dense/dense.ply
