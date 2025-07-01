import torch
from pyntcloud import PyntCloud
from typing import Dict, Union, Literal, Optional
import os
import pandas as pd

def load_splats(
    path: str,
    device: Union[str, torch.device] = 'cpu'
) -> Dict[str, torch.nn.Parameter]:
    """
    Load splat data from a .ply or .ckpt file.

    Args:
        path (str): Path to the file (.ply or .ckpt).
        device (str): Device to load tensors on. Default is 'cpu'.

    Returns:
        splats (dict): Dictionary of torch.nn.Parameter containing splat data.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.ply':
        print("loading ply file")
        cloud = PyntCloud.from_file(path)
        data = cloud.points

        # Mapping from column names in the .ply to model parameter names
        mapping = {
            "_model.gauss_params.means": ["x", "y", "z"],
            "_model.gauss_params.opacities": "opacity",
            "_model.gauss_params.quats": ["rot_0", 'rot_1', 'rot_2', 'rot_3'],
            "_model.gauss_params.scales": ["scale_0", "scale_1", "scale_2"],
            "_model.gauss_params.features_dc": [f"f_dc_{i}" for i in range(3)],
            "_model.gauss_params.features_rest": [f"f_rest_{i}" for i in range(45)]
        }

        ckpt = {"pipeline": {}}

        for key, value in mapping.items():
            if isinstance(value, list):
                ckpt["pipeline"][key] = torch.tensor(data[value].values, dtype=torch.float32)
            else:
                ckpt["pipeline"][key] = torch.tensor(data[value].values, dtype=torch.float32)

        # Reshape tensors
        ckpt["pipeline"]["_model.gauss_params.means"] = ckpt["pipeline"]["_model.gauss_params.means"].reshape(-1, 3)
        ckpt["pipeline"]["_model.gauss_params.opacities"] = ckpt["pipeline"]["_model.gauss_params.opacities"].reshape(-1)
        ckpt["pipeline"]["_model.gauss_params.scales"] = ckpt["pipeline"]["_model.gauss_params.scales"].reshape(-1, 3)
        ckpt["pipeline"]["_model.gauss_params.features_dc"] = ckpt["pipeline"]["_model.gauss_params.features_dc"].reshape(-1, 1, 3)
        ckpt["pipeline"]["_model.gauss_params.features_rest"] = ckpt["pipeline"]["_model.gauss_params.features_rest"].reshape(-1, 15, 3)

   


        # Mapping for renaming
        rename_map = {
            "_model.gauss_params.means": "means",
            "_model.gauss_params.opacities": "opacities",
            "_model.gauss_params.quats": "quats",
            "_model.gauss_params.scales": "scales",
            "_model.gauss_params.features_dc": "sh0",
            "_model.gauss_params.features_rest": "shN"
        }

        step = 0 # ply do not contain step information

        # Convert to splats dictionary
        splats = {
            new_key: torch.nn.Parameter(ckpt["pipeline"][old_key] if new_key == "sh0"
                                        else ckpt["pipeline"][old_key].squeeze() if new_key == "opacities"
                                        else ckpt["pipeline"][old_key])
            for old_key, new_key in rename_map.items()
        }

        

    elif ext == '.ckpt':
        print("loading ckpt file")
        ckpt = torch.load(path, map_location=device)

        # Mapping for renaming
        rename_map = {
            "_model.gauss_params.means": "means",
            "_model.gauss_params.opacities": "opacities",
            "_model.gauss_params.quats": "quats",
            "_model.gauss_params.scales": "scales",
            "_model.gauss_params.features_dc": "sh0",
            "_model.gauss_params.features_rest": "shN"
        }

        step = ckpt["step"]
        
        # Convert to splats dictionary
        splats = {
            new_key: torch.nn.Parameter(ckpt["pipeline"][old_key].unsqueeze(1) if new_key == "sh0"
                                        else ckpt["pipeline"][old_key].squeeze() if new_key == "opacities"
                                        else ckpt["pipeline"][old_key])
            for old_key, new_key in rename_map.items()
        }


    elif ext == '.pt':
        print("loading pt file")

        # Load and concatenate splats from checkpoints
        path = [path]
        ckpts = [
            torch.load(file, map_location=device, weights_only=True) for file in path
        ]

        param_names = ckpts[0]["splats"].keys()

        step = ckpts["step"]

        splats = {
            name: torch.nn.Parameter(
                torch.cat([ckpt["splats"][name] for ckpt in ckpts], dim=0)
            )
            for name in param_names
        }


    else:
        raise ValueError(f"Unsupported file extension: {ext}")


    return step, splats



def save_splats(
    path: str,
    data: Dict[str, Union[Dict[str, torch.nn.Parameter], int]],
    file_type: Literal['ply', 'pt', 'ckpt'] = 'ply',
    step_value: Optional[int] = None
) -> None:
    """
    Export splat data to .ply, .pt, or .ckpt format.
    
    Args:
        path (str): Output path without file extension.
        data (dict): Dictionary containing 'splats' and optionally 'step'.
        file_type (str): One of 'ply', 'pt', or 'ckpt'.
        step_value (int, optional): If provided with file_type='pt', a .ckpt is also saved.
    """
    assert file_type in ['ply', 'pt', 'ckpt'], "Unsupported file type"
    
    if file_type == 'ply':
        path += ".ply"
        splats = data["splats"]
        print("Saving to .ply format")

        export_data = {
            "x": splats["means"][:, 0].detach().cpu().numpy(),
            "y": splats["means"][:, 1].detach().cpu().numpy(),
            "z": splats["means"][:, 2].detach().cpu().numpy(),
            "opacity": splats["opacities"].detach().cpu().numpy(),
            "rot_0": splats["quats"][:, 0].detach().cpu().numpy(),
            "rot_1": splats["quats"][:, 1].detach().cpu().numpy(),
            "rot_2": splats["quats"][:, 2].detach().cpu().numpy(),
            "rot_3": splats["quats"][:, 3].detach().cpu().numpy(),
            "scale_0": splats["scales"][:, 0].detach().cpu().numpy(),
            "scale_1": splats["scales"][:, 1].detach().cpu().numpy(),
            "scale_2": splats["scales"][:, 2].detach().cpu().numpy(),
        }

        for i in range(3):
            export_data[f"f_dc_{i}"] = splats["sh0"][:, 0, i].detach().cpu().numpy()
        for i in range(45):
            c = i % 3
            j = i // 3
            export_data[f"f_rest_{i}"] = splats["shN"][:, j, c].detach().cpu().numpy()

        df = pd.DataFrame(export_data)
        cloud = PyntCloud(df)
        cloud.to_file(path)

    elif file_type == 'pt':
        pt_path = path + ".pt"
        print(f"Saving to .pt format at {pt_path}")
        torch.save(data, pt_path)

        if step_value is not None:
            print("Also generating .ckpt file from .pt content")
            _save_ckpt_from_splats(pt_path, path + ".ckpt", step_value)

    elif file_type == 'ckpt':
        print(f"Saving to .ckpt format at {path + '.ckpt'}")
        _save_ckpt_from_splats(None, path + ".ckpt", step_value, direct_data=data)


def _save_ckpt_from_splats(
    pt_path: str,
    save_path: str,
    step_value: int,
    direct_data: Optional[Dict[str, Dict[str, torch.nn.Parameter]]] = None
) -> None:
    """Internal helper to convert .pt or dict to .ckpt."""
    device = torch.device("cpu")

    if direct_data is not None:
        splats = direct_data["splats"]
    else:
        pt_data = torch.load(pt_path, map_location=device)
        splats = pt_data["splats"]

    inverse_rename_map = {
        "means": "_model.gauss_params.means",
        "opacities": "_model.gauss_params.opacities",
        "quats": "_model.gauss_params.quats",
        "scales": "_model.gauss_params.scales",
        "sh0": "_model.gauss_params.features_dc",
        "shN": "_model.gauss_params.features_rest"
    }

    pipeline = {}
    for new_key, old_key in inverse_rename_map.items():
        tensor = splats[new_key].data
        if new_key == "sh0":
            tensor = tensor.squeeze(1)
        elif new_key == "opacities":
            tensor = tensor.unsqueeze(-1)
        pipeline[old_key] = tensor

    ckpt = {
        "step": step_value,
        "pipeline": pipeline
    }

    torch.save(ckpt, save_path)
    print(f"Saved .ckpt to: {save_path}")
