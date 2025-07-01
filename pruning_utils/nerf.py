import os
import json
import torch
from typing import Any, Dict, List, Optional
import imageio.v2 as imageio
import numpy as np
import cv2

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def convert_opengl_to_colmap(R):
    M_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return R @ M_flip


class NerfParser:
    """NeRF Studio parser (transforms.json)."""

    


    def __init__(self, data_dir: str, factor: int = 1, normalize: bool = False, test_every: int = 8):


        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        # Load transforms.json
        transform_path = os.path.join(data_dir, "transforms.json")
        if not os.path.exists(transform_path):
            raise FileNotFoundError(f"Could not find transforms.json at {transform_path}")
        with open(transform_path, "r") as f:
            meta = json.load(f)

        # Store camera poses and intrinsics
        camtoworlds = []
        image_paths = []
        depth_paths = []

        Ks_dict = {}
        imsize_dict = {}
        camera_ids = []
        image_names = []
        mask_dict = {}

        fl_x = meta["fl_x"]
        fl_y = meta["fl_y"]
        cx = meta["cx"]
        cy = meta["cy"]
        w = meta["w"]
        h = meta["h"]
        fx, fy = fl_x / factor, fl_y / factor
        cx, cy = cx / factor, cy / factor
        width, height = w // factor, h // factor

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        cam_id = 0  # Single intrinsics shared across all images

        for i, frame in enumerate(meta["frames"]):
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            camtoworlds.append(c2w)

            

            rel_image_path = frame["file_path"]
            if not rel_image_path.lower().endswith((".png", ".jpg")):
                rel_image_path += ".png"  # default fallback

            image_path = os.path.join(data_dir, rel_image_path)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found.")
            image_paths.append(image_path)
            image_names.append(os.path.basename(rel_image_path))
            camera_ids.append(cam_id)



            rel_depth_path = frame.get("depth_file_path")
            if rel_depth_path is None:
                raise ValueError(f"No depth_file_path found for frame {i}")
            if not rel_depth_path.lower().endswith((".png", ".jpg", ".exr")):
                rel_depth_path += ".png"
            depth_path = os.path.join(data_dir, rel_depth_path)
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Depth image not found: {depth_path}")
            depth_paths.append(depth_path)

        camtoworlds = np.stack(camtoworlds, axis=0)
        image_paths = np.array(image_paths)
        image_names = np.array(image_names)
        depth_paths = np.array(depth_paths)

      


        # Normalize world space if requested
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)

            T2 = align_principle_axes(camtoworlds[:, :3, 3])
            camtoworlds = transform_cameras(T2, camtoworlds)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = list(image_names)
        self.image_paths = list(image_paths)
        self.depth_paths = list(depth_paths)
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = {cam_id: K}
        self.params_dict = {cam_id: np.array([], dtype=np.float32)}  # no distortion
        self.imsize_dict = {cam_id: (width, height)}
        self.mask_dict = {cam_id: None}
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.points_err = np.zeros((0,), dtype=np.float32)
        self.points_rgb = np.zeros((0, 3), dtype=np.uint8)
        self.point_indices = {}  # Not available in NeRF Studio format
        self.transform = transform

        # Resize intrinsics if actual image differs from JSON
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        s_height, s_width = actual_height / height, actual_width / width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # NeRF Studio has no distortion, so no undistortion mapping needed
        self.mapx_dict = {}
        self.mapy_dict = {}
        self.roi_undist_dict = {}

        # Estimate scene scale from camera positions
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = float(np.max(dists))


        print(f"[Parser] Loaded {len(self.image_paths)} NeRF Studio frames.")

        self.camtoworlds[:, :3, :3] = convert_opengl_to_colmap(self.camtoworlds[:, :3, :3])


class NerfDataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: NerfParser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item] 
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # Get the full depth path from the parser
            depth_filename = self.parser.depth_paths[index]

            if not os.path.exists(depth_filename):
                raise FileNotFoundError(f"Depth file not found: {depth_filename}")

            # Load depth
            depth = imageio.imread(depth_filename).astype(np.float32)

            # Apply patching if needed
            if self.patch_size is not None:
                depth = depth[y : y + self.patch_size, x : x + self.patch_size]

            if depth.shape[:2] != image.shape[:2]:
                raise ValueError(f"Depth shape {depth.shape} doesn't match image shape {image.shape}")

            data["depth"] = torch.from_numpy(depth).float()
            




            


        return data

