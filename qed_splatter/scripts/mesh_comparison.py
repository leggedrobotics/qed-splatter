import argparse
import os
import json
import numpy as np
import open3d as o3d
from open3d.t.geometry import Metric, MetricParameters
import pointcloud_alignment

def load_transforms_json(path):
    test_cam_infos = []
    train_cam_infos = []

    with open(os.path.join(path, "transforms.json")) as json_file:
        contents = json.load(json_file)
        fovx = 2 * np.arctan(contents["w"] / (2 * contents["fl_x"]))

        frames = contents["frames"]

        # Convert to set for faster evaluation
        pointcloud = o3d.t.geometry.PointCloud()

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])
            print("reading camera", cam_name)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            invdepthmap=None
            depth_path = os.path.join(path, frame["depth_file_path"])
            invdepthmap = np.load(depth_path).astype(np.float32) # load in m

            # Mask out NaN and infinite values
            mask = np.isfinite(invdepthmap)
            valid_pixels = invdepthmap[mask]

            invdepthmap = np.ascontiguousarray(invdepthmap)

            pointcloud_from_depth = o3d.t.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(invdepthmap),
                o3d.camera.PinholeCameraIntrinsic(
                    width=contents["w"],
                    height=contents["h"],
                    fx=contents["fl_x"],
                    fy=contents["fl_y"],
                    cx=contents["cx"],
                    cy=contents["cy"]
                ),
                extrinsic=w2c,  # World to camera transform
                depth_scale=1.0,  # Adjust as necessary
                depth_trunc=100.0,  # Adjust as necessary
                stride=1
            )

            pointcloud += pointcloud_from_depth
            
        pointcloud = pointcloud.voxel_down_sample(voxel_size=0.05)  # Adjust voxel size as needed
        return pointcloud

def process_dataset(dataset_path, mesh_path, nerfstudio_scale):
    """Reads and processes the dataset."""
    
    if os.path.exists(os.path.join(dataset_path, "gt_pointcloud.ply")):
        print("Found gt_pointcloud.ply file")
        pcd = o3d.t.io.read_point_cloud(os.path.join(dataset_path, "gt_pointcloud.ply"))
        pcd.point.positions = pcd.point.positions.to(o3d.core.float32)  # Convert to Float32
        
        print(f"Loaded gt point cloud with {pcd.point.positions.shape[0]} points.")
    else:
        print("No gt_pointcloud.ply file found, generating...")
        if os.path.exists(os.path.join(dataset_path, "transforms.json")):
            print("Found transforms.json file, assuming Nerfstudio data set!")
            pcd = load_transforms_json(dataset_path)
            print(f"Loaded point cloud with {pcd.point.positions.shape[0]} points.")
            # save the point cloud to a file as ply
            o3d.t.io.write_point_cloud(os.path.join(dataset_path, "gt_pointcloud.ply"), pcd)
            print(f"Saved point cloud to {os.path.join(dataset_path, 'gt_pointcloud.ply')}")
        else:
            raise ValueError("Unable to generate gt_pointgloud.ply. Unsupported dataset format or missing transforms.json file.")

    # Normalize point cloud vertices to the range [0, 1]
    # print("Normalizing point cloud vertices to the range [0, 1], original max and min are {} and {}".format(pcd.point.positions.max(), pcd.point.positions.min()))
    # pcd.point.positions = (pcd.point.positions - pcd.point.positions.min()) / (pcd.point.positions.max() - pcd.point.positions.min())

    # Load mesh and convert to tensor-based TriangleMesh
    mesh = o3d.t.io.read_triangle_mesh(mesh_path)
    # Normalize mesh vertices to the range [0, 1]
    mesh.vertex.positions = mesh.vertex.positions.to(o3d.core.float32)  # Convert to Float32
    print("Number of vertices in mesh: ", mesh.vertex.positions.shape[0])
    # print("Normalizing mesh vertices to the range [0, 1], original max and min are {} and {}".format(mesh.vertex.positions.max(), mesh.vertex.positions.min()))
    # mesh.vertex.positions = (mesh.vertex.positions - mesh.vertex.positions.min()) / (mesh.vertex.positions.max() - mesh.vertex.positions.min())

    center = o3d.core.Tensor([0,0,0])
    pcd_scaled = pcd.scale(nerfstudio_scale, center)

    # Filter out values that are too high   
    # Define the distance threshold
    threshold = 10.0  # Adjust this as needed

    # Compute the Euclidean distance from the origin
    distances = np.linalg.norm(pcd_scaled.point.positions.numpy(), axis=1)

    # Create a mask for points within the threshold
    mask = distances < threshold

    indices = np.where(mask)[0]

    # Apply the mask to filter points
    filtered_pcd = pcd_scaled.select_by_index(o3d.core.Tensor(indices))
    mesh_alignment_pcd = mesh.sample_points_uniformly(number_of_points=10000)  # Sample points from the mesh

    # Align the point cloud with the mesh using ICP
    target_down, target_fpfh = pointcloud_alignment.preprocess_point_cloud(filtered_pcd, voxel_size=0.05)
    source_down, source_fpfh = pointcloud_alignment.preprocess_point_cloud(mesh_alignment_pcd, voxel_size=0.05)

    # pcd_down = filtered_pcd.voxel_down_sample(voxel_size=0.05)
    ransac_result = pointcloud_alignment.execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size=0.05)
    refined_result = pointcloud_alignment.refine_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size=0.05, ransac_result=ransac_result)

    print("Drawing result")
    pointcloud_alignment.draw_registration_result(mesh_alignment_pcd, filtered_pcd, refined_result.transformation)

    aligned_mesh_pcd = mesh_alignment_pcd.transform(refined_result.transformation)
    metric_params = MetricParameters()
    metrics = aligned_mesh_pcd.compute_metrics(
        filtered_pcd, [Metric.ChamferDistance],
    metric_params)

    print("Chamfer Distance ", metrics)

def main():
    parser = argparse.ArgumentParser(description="Process a dataset.")
    parser.add_argument("--data", required=True, help="Path to the dataset")
    parser.add_argument("--mesh", required=True, help="Path to the mesh to evaluate")
    parser.add_argument("--nerfstudio-scale", type=float, default=0.04169970387999055, help="The scaling factor used in the generation of the mesh by nerfstudio. Default is 0.04169970387999055.")
    args = parser.parse_args()
    process_dataset(args.data, args.mesh, args.nerfstudio_scale)

if __name__ == "__main__":
    main()
