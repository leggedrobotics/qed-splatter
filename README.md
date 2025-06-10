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

Currently two scripts are usable:

### RGB_hard_pruner
```
python3 RGB_hard_pruner.py default --data-dir datasets/park --ckpt results/park/step-000029999.ckpt --pruning-ratio 0.1 --result-dir output

--eval-only (only evaluates, no saving, no pruning)  
--pruning-ratio 0.0 (no pruning, saved in new format)  
--output-format (ply (default), ckpt (nerfstudio), pt (gsplat))
```
### depth_hard_pruner

Works analogously to the RGB hard pruner, not all features are available.

### Soft Pruner (WIP)

To use the soft pruner during qed traing use the following command:
```
PRUNING_RATIO=0.2 SOFT_PRUNING=True DATA_DIR=<data> ns-train qed-splatter --data <data>
```
#### known issues
The current sof pruner uses ENV variables. This will be adressed later.
For the Park scene it tries to generate black gaussians to cover the sky. The enitre scene is encased in these gaussians.
