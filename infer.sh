#!/usr/bin/env bash
set -e

COMMON_ARGS=(
  configs/infer.yaml
  infer.dataset.data_root=example
  infer.dataset.split_json=dataLoader/example_split.json
  infer.dataset.projector_cfg_path=dataLoader/config.yaml
  infer.eval_novel_view_only=True
  save_slice=True
)

CUDA_VISIBLE_DEVICES=0 python inference.py \
  configs/ILV_6v.yaml \
  "${COMMON_ARGS[@]}" \
  n_views=6 \
  infer.ckpt_path=checkpoints/ckpt/6v.pth \
  infer.metric_path=outputs/metrics/6v_example.json \
  infer.save_folder=outputs/images/6v_example

CUDA_VISIBLE_DEVICES=0 python inference.py \
  configs/ILV_8v.yaml \
  "${COMMON_ARGS[@]}" \
  n_views=8 \
  infer.ckpt_path=checkpoints/ckpt/8v.pth \
  infer.metric_path=outputs/metrics/8v_example.json \
  infer.save_folder=outputs/images/8v_example

CUDA_VISIBLE_DEVICES=0 python inference.py \
  configs/ILV_10v.yaml \
  "${COMMON_ARGS[@]}" \
  n_views=10 \
  infer.ckpt_path=checkpoints/ckpt/10v.pth \
  infer.metric_path=outputs/metrics/10v_example.json \
  infer.save_folder=outputs/images/10v_example

CUDA_VISIBLE_DEVICES=0 python inference.py \
  configs/ILV_24v.yaml \
  "${COMMON_ARGS[@]}" \
  n_views=24 \
  infer.ckpt_path=checkpoints/ckpt/24v.pth \
  infer.metric_path=outputs/metrics/24v_example.json \
  infer.save_folder=outputs/images/24v_example

CUDA_VISIBLE_DEVICES=0 python inference.py \
  configs/ILV_10v_small.yaml \
  "${COMMON_ARGS[@]}" \
  n_views=10 \
  infer.ckpt_path=checkpoints/ckpt/10v_small.pth \
  infer.metric_path=outputs/metrics/10v_small_example.json \
  infer.save_folder=outputs/images/10v_small_example
