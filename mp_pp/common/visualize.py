from typing import Union, Dict
import matplotlib.pyplot as plt


def viz_loggings(
        logging: Dict,
        save_path: Union[None, str] = None
):
    """
    Visualize loggings
    :param logging: should be in dict:
    {
        "train": {
            ...
        }
        "val": {
            ...
        }
    }
    :param save_path: folder path to save figure
    :return:
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    for i, (phase_key, phase_val) in enumerate(logging.items()):
        axs[i].set_title(phase_key)
        for log_key, log_val in phase_val.items():
            axs[i].plot(log_val, label=log_key)
            axs[i].legend(loc="upper right")
    if save_path:
        plt.savefig(f"{save_path}/training_curve.png")
    else:
        plt.show()
