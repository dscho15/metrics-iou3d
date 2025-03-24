import torch

def get_predicted_boxes(
        center: torch.FloatTensor, 
        center_z: torch.FloatTensor, 
        dim: torch.FloatTensor, 
        spatial_indices: torch.FloatTensor,
        rot: torch.FloatTensor,
        feature_map_stride: int = 8,
        voxel_size: tuple = (0.1, 0.1, 0.1),
        pointcloud_range: tuple = (0, 0, 0, 5, 5, 5),
) -> torch.FloatTensor:
    
    dim = torch.exp(torch.clamp(dim, min=-5, max=5))

    rot_cos = rot[:, 0].unsqueeze(dim=1)
    rot_sin = rot[:, 1].unsqueeze(dim=1)

    angle = torch.atan2(rot_sin, rot_cos)

    xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * feature_map_stride * voxel_size[0] + pointcloud_range[0]
    ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * feature_map_stride * voxel_size[1] + pointcloud_range[1]

    box_part_list = [xs, ys, center_z, dim, angle]
    pred_box = torch.cat((box_part_list), dim=-1)
    
    return pred_box
