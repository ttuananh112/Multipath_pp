import torch


def get_pseudo_prediction(y, reg, device):
    """
    This function create pseudo ground-truth for classification
    and get regression prediction nearest to ground-truth (y_hat)
    :param y: label with shape (batch, future_steps, dim)
    :param reg: output of regression head
    with shape (batch, num_anchor, future_steps, 4)
    :param device: device
    :return:
        - pseudo_gt_idx
        - y_hat
    """
    batch_size, future_steps, dimension = y.shape
    _y = y.view(batch_size, 1, future_steps, dimension)

    # get norm2 distance
    diff = reg[:, :, :, :2] - _y
    dist = torch.linalg.norm(diff, dim=-1)
    # get sum by future_steps
    sum_dist = torch.sum(dist, dim=-1)
    pseudo_gt_idx = torch.argmin(sum_dist, dim=-1)

    # get prediction that closest to gt
    idx = torch.stack([
        torch.arange(batch_size, device=device),
        pseudo_gt_idx
    ], dim=0).tolist()
    y_hat = reg[idx]  # (batch, future_steps, 4)

    return pseudo_gt_idx, y_hat


def get_device(device: str):
    """
    Get device
    :param device:
    :return:
    """
    if "cpu" not in device:
        device = "cuda:0"  # temporal solution
    return device
