# QED-Splatter
QED-Splatter (short for Quick and Easy Depth Splatter) is a custom depth-supervised implementation of the Gaussian Splatting method, built on top of Nerfstudio. 
It is designed to provide a flexible and efficient framework for neural 3D scene reconstruction using depth information, particularly in outdoor and forested environments. 
This repository was developed as part of a [Bachelor Thesis](https://github.com/leggedrobotics/forest-digital-twin) on applying Gaussian Splatting to forest environments.

## Installation
Before installing QED-Splatter, make sure you have installed Nerfstudio following these [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html).
```
conda activate nerfstudio
cd QED-Splatter
pip install -e .
ns-install-cli
```

## Setup
Per default the `depth_unit_scale_factor` is set to millimeters. To update this change the `depth_unit_scale_factor: float` in the [dataparser](qed_splatter/dataparser.py).

The scaling to the 1 by 1 cube of nerfstudio is also enabled. For my thesis for example this needs to be disabled. To reenable this change add / comment in following lines in the [dataparser](qed_splatter/dataparser.py):
```python
auto_scale_poses: bool = False
center_method: Literal["poses", "focus", "none"] = "none"
orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
```

Lastly when using unscaled scenes, the model currently doesn't initialize the gaussians based on the depth images. There are three options here.
1. Use the default cube initialization and modify the scale of the cube with `random_scale: float = 100.0` in the [model config](qed_splatter/model.py).
2. Use the 3d Points initialization provided by splatfacto.
3. Add this functionality to the repository and submit a pull-request.

## Running the new method
To train the new method, use the following command:
```
ns-train qed-splatter --data [PATH]
```

## Pruning Extension

The pruning extension provides tools to reduce the number of Gaussians to increase the rendering speed. There are two main types of pruner available. Soft pruner reduce the number of Gaussians during training. Hard pruner are a post processing tool which can be used after training. Each pruner computes a pruning score to rank the importance of Gaussians. The least important are pruned.

Currently two scripts are usable:

### RGB_hard_pruner
This pruner uses RGB loss to compute a pruning score to do hard pruning.
```
python3 RGB_hard_pruner.py default --data-dir datasets/park --ckpt results/park/step-000029999.ckpt --pruning-ratio 0.1 --result-dir output

--eval-only (only evaluates, no saving, no pruning)  
--pruning-ratio 0.0 (no pruning, saved in new format)  
--output-format (ply (default), ckpt (nerfstudio), pt (gsplat))
```

## 📥 Required Arguments

| Argument          | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `default`         | Specifies the run configuration.   |
| `--data-dir`      | Path to the directory containing `transforms.json` (camera poses and intrinsics) and images (RGB, Depth)· |
| `--ckpt`          | Path to the pretrained model checkpoint (e.g., `results/park/step-XXXXX.ckpt`). |
| `--pruning-ratio` | Float between `0.0` and `1.0`. Proportion of the model to prune. Example: `0.1` = keep 90%. |
| `--result-dir`    | Directory where the output (pruned model) will be saved.                |


### 🔧 `transforms.json` File

This file must include the following:

- Intrinsic camera parameters:
  - `"fl_x"`, `"fl_y"`: focal lengths
  - `"cx"`, `"cy"`: principal point
  - `"w"`, `"h"`: image dimensions

- A list of frames, each containing:
  - `file_path`: path to the RGB image (relative to `your_dataset/`)
  - `depth_file_path`: path to the depth map (relative to `your_dataset/`)
  - `transform_matrix`: 4x4 camera-to-world matrix

**Example:**
```json
{
    "w": 1920,
    "h": 1080,
    "fl_x": 2198.997802734375,
    "fl_y": 2198.997802734375,
    "cx": 960.0,
    "cy": 540.0,
    "k1": 0,
    "k2": 0,
    "p1": 0,
    "p2": 0,
  "frames": [
    {
      "file_path": "images/frame_0000.png",
      "depth_file_path": "depths/frame_0000.png",
      "transform_matrix": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    },
    {
      "file_path": "images/frame_0001.png",
      "depth_file_path": "depths/frame_0001.png",
      "transform_matrix": [[0,2,0,0], [0,1,3,0], [0,0,1,0], [0,5,0,1]]
    }
  ]
}
```

## Input Format

The code supports multiple output formats. The format is detected automatically. 
- `ply` : expects a Nerfstudio format for the transforms. 
- `ckpt` : expects a Nerfstudio format for the transforms. 
- `pt` : expects a gsplat format for the transforms. 


### Depth_hard_pruner
This pruner uses depth loss to compute a pruning score to do hard pruning. It works analogously to the RGB hard pruner but not all features are available.
```
python3 depth_hard_pruner.py default --data-dir datasets/park --ckpt results/park/step-000029999.ckpt --pruning-ratio 0.1 --result-dir output

--eval-only (only evaluates, no saving, no pruning)  
--pruning-ratio 0.0 (no pruning, saved in new format)  
--output-format (ply (default), ckpt (nerfstudio), pt (gsplat))
```

#### Known Issues
For the Park scene it tries to generate black gaussians to cover the sky. The enitre scene is encased in these gaussians.
