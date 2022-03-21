import json
import math
import numpy as np
from typing import List, Dict, Tuple

import pandas as pd
import torch
from torch.nn import functional as F

np.seterr(divide='ignore', invalid='ignore')


# --- common function ---
def rotate(
        x: float,
        y: float,
        angle: float
) -> Tuple[float, float]:
    """
    Rotate point with angle
    :param x: x-coordinate
    :param y: y-coordinate
    :param angle: angle heading
    :return:
        Tuple[float, float]
            - rotated_x
            - rotated_y
    """
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def rotate_vector(
        obj: np.ndarray,
        agent: np.ndarray,
        angle: float
) -> np.ndarray:
    """
    Rotate vector obj with angle
    where, agent is origin

    :param obj: (N, 2)
    :param agent: (2,)
    :param angle: (1,)
    :return:
    """
    x = obj[:, 0] - agent[0]
    y = obj[:, 1] - agent[1]
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return np.stack([res_x, res_y], axis=-1)


def distance(
        a: np.ndarray,
        b: Tuple[float, np.ndarray] = 0,
        axis: int = -1
):
    """
    Length of vector a
    or distance of 2 points (a, b)
    :param a:
    :param b:
    :param axis:
    :return:
    """
    return np.linalg.norm(a - b, axis=axis)


def get_angle(
        v1: np.ndarray,
        v2: np.ndarray
):
    """
    Angle between 2 vectors
    :param v1:
    :param v2:
    :return:
    """
    dot_prod = np.sum(v1 * v2, axis=-1)
    cos_alpha = dot_prod / (distance(v1) * distance(v2))
    # hot-fix for waypoint==agent_pos
    if np.isnan(cos_alpha[0]):
        cos_alpha[0] = cos_alpha[1]
    return np.arccos(cos_alpha)


def convert_status_to_np(
        df_status: pd.DataFrame,
        key: str
) -> np.ndarray:
    """
    Convert status column into numpy array
    :param df_status:
    :param key: key string of status
    :return:
    """
    def __get_value(x):
        return json.loads(x)[key]

    df_status = df_status.apply(__get_value)
    return df_status.to_numpy()


def standardize_angle(
        rad_angle: np.ndarray
) -> np.ndarray:
    """
    Function to standardize radian angle,
    assure to be in range [-pi, pi]
    :param rad_angle: radian angle in numpy (N,)
    :return:
    """
    idx = rad_angle > 2 * math.pi
    # if rad_angle go over 1 round
    rad_angle[idx] = rad_angle[idx] % (2 * math.pi)

    # if rad_angle go over half round
    idx = abs(rad_angle) > math.pi
    sign = np.sign(rad_angle[idx])
    redundant = abs(rad_angle[idx]) - math.pi
    rad_angle[idx] = -sign * (math.pi - redundant)
    return rad_angle


# --- function to support dataset ---
def get_largest_num_objects(x: List[Dict]):
    """
    Get the largest number of objects in batch
    :param x: Batch data
    :return:
        (int)
    """
    largest = 0
    for sample, _ in x:
        # (batch, num_objects, history_step, dimension)
        objects = sample["others"]

        num_objects = len(objects)
        if num_objects > largest:
            largest = num_objects
    return largest


def pad_objects(x: List[Tuple]):
    """
    Pad zeros to fulfill other objects
    :param x:
    :return:
    """
    largest_num_objects = get_largest_num_objects(x)
    # loop through each object in batch
    for i in range(len(x)):
        # handling objects
        num_objects = len(x[i][0]["others"])
        if num_objects == 1:
            # (1, history_steps, dimension)
            data_tmp = x[i][0]["others"][0].unsqueeze(0)
        else:
            # (num_objects, history_step, dimension)
            data_tmp = torch.stack(x[i][0]["others"], dim=0)

        # permute dim num_object to last
        data_tmp = torch.permute(data_tmp, (1, 2, 0))
        # padding zeros to num_objects dim
        pad = (0, largest_num_objects - num_objects)
        data_tmp = F.pad(data_tmp, pad)
        # permute back
        data_tmp = torch.permute(data_tmp, (2, 0, 1))
        # assign value
        x[i][0]["others"] = data_tmp
