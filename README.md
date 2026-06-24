<div align="center">
<h1 align="center">
ILV: Iterative Latent Volumes for Fast and Accurate <br> Sparse-View CT Reconstruction
</h1>

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://sngryonglee.github.io/ILV/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.14915-b31b1b.svg)](https://arxiv.org/abs/2603.14915)
</div>

Official implementation of **ILV: Iterative Latent Volumes for Fast and Accurate Sparse-View CT Reconstruction**.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/sngryongLee/ILV.git
cd ILV

# 2. Create and activate a conda environment
conda create -n ilv python=3.11 -y
conda activate ilv

# 3. Install PyTorch (adjust PyTorch and CUDA version to match your system)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Build the CUDA rasterization/voxelization extension
pip install -e third_party/xray-gaussian-rasterization-voxelization --no-build-isolation
```

## Checkpoints

Download pretrained checkpoints from [Hugging Face](https://huggingface.co/sngryong/ILV/tree/main), then place them under `checkpoints/ckpt/`:

```text
checkpoints/
└── ckpt/
    ├── 6v.pth
    ├── 8v.pth
    ├── 10v.pth
    ├── 24v.pth
    └── 10v_small.pth
```

## Data

The data directory should follow the same structure as `example/`: each scene is placed under a dataset folder, and each scene contains a CT volume and precomputed projections.

```text
example/
├── amos/
│   └── amos_5387/
│       ├── volume.nii.gz
│       └── projections.npy
├── rsna2023/
│   └── 1061_48417/
│       ├── volume.nii.gz
│       └── projections.npy
└── ...
```

Dataset splits are JSON files with `train` and `test` entries:

```json
{
  "train": [
    {"name": "case_001", "path": "dataset/case_001"}
  ],
  "test": [
    {"name": "amos_5387", "path": "amos/amos_5387"}
  ]
}
```

The default config files use:

```text
dataLoader/split.json
dataLoader/config.yaml
dataLoader/camera_params.npz
```

## Inference

Before running inference, make sure the checkpoints are downloaded and placed under `checkpoints/ckpt/`.
The default `infer.sh` runs inference on the sample cases in `example/`.

To run inference on the example split:

```bash
bash infer.sh
```

This runs the 6-view, 8-view, 10-view, 24-view, and 10-view-small models and saves results under:

```text
outputs/images/
outputs/metrics/
```

You can also run a single model manually:

```bash
python inference.py \
  configs/ILV_10v.yaml \
  configs/infer.yaml \
  n_views=10 \
  infer.dataset.data_root=example \
  infer.dataset.split_json=dataLoader/example_split.json \
  infer.dataset.projector_cfg_path=dataLoader/config.yaml \
  infer.ckpt_path=checkpoints/ckpt/10v.pth \
  infer.metric_path=outputs/metrics/10v_example.json \
  infer.save_folder=outputs/images/10v_example \
  save_slice=True \
  save_volume=True
```

## Training

Update the dataset paths in the config files, then run:

```bash
python train.py \
  configs/ILV_10v.yaml \
  data_root=/path/to/processed_data \
  split_json=/path/to/split.json \
  projector_cfg_path=/path/to/config.yaml
```

By default, `train.py` loads `configs/ILV_10v.yaml` when no config path is provided.

Available model configs include:

```text
configs/ILV_6v.yaml
configs/ILV_8v.yaml
configs/ILV_10v.yaml
configs/ILV_24v.yaml
configs/ILV_10v_small.yaml
```

## Acknowledgement

This project uses a CUDA rasterization/voxelization extension adapted from the CUDA rasterizer in [R<sup>2</sup>-Gaussian](https://github.com/ruyi-zha/r2_gaussian).

## Citation

If you find this work useful, please cite:

```bibtex
@article{lee2026ilv,
  title={ILV: Iterative Latent Volumes for Fast and Accurate Sparse-View CT Reconstruction},
  author={Lee, Seungryong and Baek, Woojeong and Lee, Joosang and Park, Eunbyung},
  journal={arXiv preprint arXiv:2603.14915},
  year={2026}
}
```
