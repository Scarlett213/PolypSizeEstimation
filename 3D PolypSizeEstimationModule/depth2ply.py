import os
import open3d as o3d
import torch
import time
import PIL.Image as pil




def clean(path):
    start = time.time()
    "read cloud"
    cloud1 = o3d.io.read_point_cloud(path)

    colors = np.asarray(cloud1.colors)
    points = np.asarray(cloud1.points)
    idx = np.nonzero(colors.any(axis=1))[0]
    colors = colors[idx, :]
    points = points[idx, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    "save cloud"
    o3d.io.write_point_cloud(path, pcd)
    end = time.time()


def plot_growing_cloud(depth_path, img_path, mask_path, intrinsics):

    mask = pil.open(mask_path).convert('L')
    depth0 = np.load(depth_path)/10  #in cm
    mask = pil.open(mask_path).convert('RGB')
    mask = np.array(mask) / 255
    im0 = pil.open(img_path).convert('RGB')
    im0 = np.array(im0) / 255
    im0 = im0 * mask
    im0 = im0[:, :, :3].reshape((-1, 3))
    rgbs = im0[:, :]
    cloud_rgbs = rgbs
    img_height = depth0.shape[0]
    img_width = depth0.shape[1]
    cam_coords0 = pixel2cam(torch.tensor(depth0.reshape((1, img_height, img_width))).float(),
                            torch.tensor(intrinsics).inverse().float())
    cam_coords_flat0 = cam_coords0.reshape(1, 3, -1).numpy()
    cloud_gt0 = np.squeeze(cam_coords_flat0)
    return cloud_gt0, cloud_rgbs


def pixel2cam(depth, intrinsics_inv):
    b, h, w = depth.size()
    pixel_coords = set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def set_id_grid(depth):
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)
    return torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    depth_path = '' # npy format
    img_path = '' # png format
    mask_path = '' # png format
    ply_path = '' # ply format
    intrinsics = np.eye(3)
    intrinsics[0, 0] = 1.0 # fx
    intrinsics[0, 2] = 0.5 #cx
    intrinsics[1, 1] = 1.0 # fy
    intrinsics[1, 2] = 0.5 #cy
    cloud_gts, cloud_rgb = plot_growing_cloud(depth_path, img_path, mask_path, intrinsics)
    cloud_gts = np.array(cloud_gts)
    cloud_gts = cloud_gts.transpose(1, 0)
    cloud_rgb = np.array(cloud_rgb)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_gts)
    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb)
    pcd = o3d.geometry.PointCloud.remove_non_finite_points(pcd)
    o3d.io.write_point_cloud(ply_path, pcd)
    clean(ply_path)
