import torch
import numpy as np
import random


def set_seed(
        number: int = 0
) -> None:
    """
    Set seed random number
    :param number: seed number
    :return:
    """
    # random
    random.seed(number)
    # numpy
    np.random.seed(number)
    # torch
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    torch.cuda.manual_seed_all(number)
