#!/bin/bash
scene_name=$1

python reconstruction.py --scene_name ${scene_name} --version gim

# gim
colmap image_undistorter \
    --image_path inputs/${scene_name}/images \
    --input_path outputs/${scene_name}/gim_dkm/sparse \
    --output_path outputs/${scene_name}/gim_dkm/dense

colmap patch_match_stereo \
    --workspace_path outputs/${scene_name}/gim_dkm/dense

colmap stereo_fusion \
    --workspace_path outputs/${scene_name}/gim_dkm/dense \
    --output_path outputs/${scene_name}/gim_dkm/dense/dense.ply
