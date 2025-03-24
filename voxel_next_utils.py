import torch

def get_predicted_boxes(
        center: torch.FloatTensor, 
        center_z: torch.FloatTensor, 
        dim: torch.FloatTensor, 
        spatial_indices: torch.FloatTensor,
        rot: torch.FloatTensor,
        feature_map_stride: int,
        voxel_size: tuple,
        pointcloud_range: tuple,
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

def rotate_class_specific_nms_iou(self, boxes: torch.FloatTensor, scores: torch.FloatTensor, iou_preds: torch.FloatTensor, labels: torch.LongTensor, rectifier: torch.FloatTensor, nms_configs):
    """
    :param boxes: (N, 5) [x, y, z, l, w, h, theta]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert isinstance(rectifier, list)

    box_preds_list, scores_list, labels_list = [], [], []
    for cls in range(self.num_class):
        mask = labels == cls
        boxes_cls = boxes[mask]
        scores_cls = torch.pow(scores[mask], 1 - rectifier[cls]) * torch.pow(iou_preds[mask].squeeze(-1), rectifier[cls])
        labels_cls = labels[mask]

        selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=scores_cls, box_preds=boxes_cls, 
                                                    nms_config=nms_configs[cls], score_thresh=None)

        box_preds_list.append(boxes_cls[selected])
        scores_list.append(scores_cls[selected])
        labels_list.append(labels_cls[selected])

    return torch.cat(box_preds_list, dim=0), torch.cat(scores_list, dim=0), torch.cat(labels_list, dim=0)
