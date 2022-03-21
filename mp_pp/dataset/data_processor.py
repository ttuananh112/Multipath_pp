import torch
import json
import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Union, List, Tuple, Dict
from omegaconf import DictConfig
from dataset.carla_helper import MapHelper
from dataset.exception import FailedProcessing

from dataset.utils import (
    distance, rotate_vector, get_angle,
    convert_status_to_np, standardize_angle,
    pad_objects
)


class DataProcessor:
    """
    This main objective of this class
    is to process input and output data before feeding into model
    """

    def __init__(
            self,
            configs: DictConfig,
            map_path: str
    ):
        self._configs = configs
        # params
        self._history_steps = configs.model.data.history_steps
        self._future_steps = configs.model.data.future_steps
        self._num_waypoints = configs.model.data.num_waypoints
        self._dim = configs.model.data.dim
        # map_helper
        self._map_helper = MapHelper(map_path=map_path)
        # init current agent's state
        self._pos_agent, self._agent_heading = None, None

    def __process_input(
            self,
            df: pd.DataFrame
    ) -> Dict:
        """
        Main function to process input data
        :param df: data in DataFrame
        :return:
            (Dict): all value are normalized by agent's orientation
                - map: (num_waypoints, dim)
                - traffic_light: (dim, )
                - agent: (dim, )
                - others: List[(history_steps, dim)]
        """
        map = self.__process_map()
        traffic_light = self.__process_traffic_light(df)
        agent = self.__process_dynamics(df, object_type="AGENT")
        others = self.__process_dynamics(df, object_type="OTHERS")

        return {
            "map": map,
            "traffic_light": traffic_light,
            "agent": agent,
            "others": others
        }

    def __process_output(
            self,
            df: pd.DataFrame
    ) -> torch.Tensor:
        """
        Main function to process output data
        :param df: data in DataFrame
        :return:
            (torch.Tensor): (future_steps, 2) normalized by agent's orientation
        """
        # get AGENT data
        df_agent = df.loc[df["object_type"] == "AGENT"]
        if len(df_agent) != self._future_steps:
            raise FailedProcessing

        _x = df_agent["center_x"].to_numpy()
        _y = df_agent["center_y"].to_numpy()
        _pos = np.stack([_x, _y], axis=-1)
        # agent-orientation normalize
        _normed_pos = rotate_vector(_pos, self._pos_agent, self._agent_heading)
        torch_gt = torch.from_numpy(_normed_pos)
        return torch_gt

    def __process_map(self):
        """
        Function to process waypoint map using self._map_helper
        Positions are normalized by agent's orientation
        :return:
            (torch.Tensor): (num_waypoints, dim)
                [0]: pos_x
                [1]: pos_y
                [2]: dist_from_agent_x
                [3]: dist_from_agent_y
                [4]: diff_heading_to_agent
        """
        # get local waypoints
        _, polygons = self._map_helper.get_local_lanes(
            agent_x=self._pos_agent[0], agent_y=self._pos_agent[1],
            heading=self._agent_heading
        )

        # terminate when can not find waypoint
        if len(polygons) == 0:
            raise FailedProcessing

        waypoints = np.concatenate(polygons, axis=0)  # (num_wp, 2)

        # get N-nearest waypoints
        dist_wp_agent = distance(waypoints, self._pos_agent)
        idx_min_dist = dist_wp_agent.argsort()[:self._num_waypoints]

        # pos
        pos_wp = waypoints[idx_min_dist]  # (self._num_waypoints, 2)
        # rotate wp by agent's orientation
        normed_pos_wp = rotate_vector(pos_wp, self._pos_agent, self._agent_heading)
        # dist
        # (self._num_waypoints, 1)
        dist_wp = np.expand_dims(distance(pos_wp, self._pos_agent), axis=-1)
        # diff-angle
        v_agent_wp = pos_wp - self._pos_agent
        v_x_unit = np.tile([1, 0], len(v_agent_wp)).reshape(-1, 2)
        wp_angle_heading = get_angle(v_agent_wp, v_x_unit)
        # (self._num_waypoints, 1)
        diff_angle = np.expand_dims(
            np.abs(self._agent_heading) - wp_angle_heading,
            axis=-1
        )

        # gather all information
        wp_data = np.concatenate([normed_pos_wp, dist_wp, diff_angle], axis=-1)

        # pad zeros
        wp_data = torch.from_numpy(wp_data)
        num_wp, dim_wp = wp_data.shape
        torch_wp = torch.zeros((self._num_waypoints, self._dim))
        torch_wp[:num_wp, :dim_wp] = wp_data  # (self._num_waypoints, self._dim)

        return torch_wp

    def __process_traffic_light(
            self,
            df: pd.DataFrame
    ) -> torch.Tensor:
        """
        Function to process traffic light
        Positions are normalized by agent's orientation
        :param df: data in DataFrame
        :return:
            (torch.Tensor): (dim, )
                [0]: pos_x
                [1]: pos_y
                [2]: is_red
                [3]: is_yellow
                [4]: is_green
        """
        light_mapping = {
            "RED": 0,
            "YELLOW": 1,
            "GREEN": 2
        }
        # get current state of traffic light
        row = df.loc[df["type"] == "traffic_light"]
        torch_tl = torch.zeros(self._dim)

        # encode only if there is traffic light...
        # else, all-zeros
        if len(row) > 0:
            row = row.iloc[-1]
            # get traffic light data
            tl_x = row["center_x"]
            tl_y = row["center_y"]
            tl_pos = np.array([[tl_x, tl_y]])
            tl_stt = json.loads(row["status"])["light_state"]

            # normalize traffic light position
            normed_tl_pos = rotate_vector(tl_pos, self._pos_agent, self._agent_heading).squeeze()
            # one-hot encoding traffic light status
            encode_stt = np.zeros(3)
            encode_stt[light_mapping[tl_stt]] = 1

            tl_data = np.concatenate([normed_tl_pos, encode_stt])
            tl_data = torch.from_numpy(tl_data)
            # pad zeros
            torch_tl[:len(tl_data)] = tl_data  # (self._dim)

        return torch_tl

    def __process_dynamics(
            self,
            df: pd.DataFrame,
            object_type: str
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Process state of dynamic object
        including AGENT and OTHERS
        :param df: data in DataFrame
        :param object_type: should be in ["AGENT", "OTHERS"]
        :return:
            Could be List[torch.Tensor] or torch.Tensor
                - List[torch.Tensor]: for OTHERS
                (num_others, history_steps, dim)
                - torch.Tensor: for AGENT
                (history_steps, dim)
        """

        def __process_each_object(df_group_by_object):
            # get data
            _x = df_group_by_object["center_x"].to_numpy()
            _y = df_group_by_object["center_y"].to_numpy()
            _heading = df_group_by_object["heading"].to_numpy()
            _mag_vel = convert_status_to_np(df_group_by_object["status"], key="velocity")
            _diff_vel = np.diff(_mag_vel)
            _mag_acc = np.concatenate([[_diff_vel[0]], _diff_vel])
            _diff_heading = standardize_angle(np.diff(_heading))
            _turn_rate = np.concatenate([[_diff_heading[0]], _diff_heading])

            # --- normalize data ---
            # position
            _pos = np.stack([_x, _y], axis=-1)
            _normed_pos = rotate_vector(_pos, self._pos_agent, self._agent_heading)  # (self._history_steps, 2)
            # heading
            _normed_heading = np.expand_dims(_heading - self._agent_heading, axis=-1)  # (self._history_steps, 1)
            # velocity
            _vel_x = _mag_vel * np.cos(_heading)
            _vel_y = _mag_vel * np.sin(_heading)
            _vel = np.stack([_vel_x, _vel_y], axis=-1)
            _normed_vel = rotate_vector(_vel, self._pos_agent, self._agent_heading)  # (self._history_steps, 2)
            # acceleration
            _acc_x = _mag_acc * np.cos(_heading)
            _acc_y = _mag_acc * np.sin(_heading)
            _acc = np.stack([_acc_x, _acc_y], axis=-1)
            _normed_acc = rotate_vector(_acc, self._pos_agent, self._agent_heading)  # (self._history_steps, 2)
            # turn rate
            _normed_turn_rate = np.expand_dims(_turn_rate, axis=-1)  # (self._history_steps, 1)
            # gather information
            dynamic_data = np.concatenate([
                _normed_pos,
                _normed_vel,
                _normed_acc,
                _normed_heading,
                _normed_turn_rate
            ], axis=-1)
            dynamic_data = torch.from_numpy(dynamic_data)
            h_data, w_data = dynamic_data.shape

            # pad zeros
            torch_dynamic = torch.zeros((self._history_steps, self._dim))
            torch_dynamic[:h_data, :w_data] = dynamic_data
            return torch_dynamic

        # --- main flow ---
        if object_type not in ["AGENT", "OTHERS"]:
            raise FailedProcessing

        # AGENT
        if object_type == "AGENT":
            df_by_object = df.loc[df["object_type"] == "AGENT"]
            return __process_each_object(df_by_object)
        # OTHERS
        container = list()
        df_others = df.loc[df["object_type"] == "OTHERS"]
        df_group_by_id = df_others.groupby(by=["id"])
        for _, data in df_group_by_id:
            container.append(__process_each_object(data))
        return container

    def __get_inp_out_data(
            self,
            df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process to separate input/output DataFrame
        :param df: all DataFrame
        :return:
            Tuple[pd.DataFrame, pd.DataFrame]
                - input: with number of timestamps = history_step
                - output: with number of timestamps = future_step
        """
        df_by_ts = df.groupby(by=["timestamp"])
        # store inp/out data separately
        inp_data = pd.DataFrame(columns=df.columns)
        out_data = pd.DataFrame(columns=df.columns)
        for i, (ts, df_tick) in enumerate(df_by_ts):
            if i < self._history_steps:
                inp_data = pd.concat([inp_data, df_tick])
            else:
                out_data = pd.concat([out_data, df_tick])
        return inp_data, out_data

    @staticmethod
    def __get_current_agent_state(
            df: pd.DataFrame
    ) -> Tuple[np.ndarray, float]:
        """
        Get current state of AGENT
        :param df: input data in DataFrame
        :return:
            (Tuple[np.ndarray, float]):
                - position of agent in world coordinate: (agent_x, agent_y)
                - heading of agent in world coordinate
        """
        row = df.loc[df["object_type"] == "AGENT"].iloc[-1]
        return (
            np.array([row["center_x"], row["center_y"]]),
            row["heading"]
        )

    def process(
            self,
            inputs: Union[str, pd.DataFrame],
            is_inference=False
    ) -> Union[
        Tuple[Dict, torch.Tensor],
        Tuple[None, None]
    ]:
        """
        Main function to process data
        :param inputs: could be:
            - path to csv file or
            - DataFrame
        :param is_inference: flag to trigger inference
            - if False:
                + inputs should include timestamps for input and output
                + process output data (for training/validating process)
            - else True:
                + inputs only include timestamps for input
                + do not process output data

        :return:
            Tuple[Dict, torch.Tensor]
                - processed input data
                - processed output data
            if return (None, None) -> process failed

        """
        assert isinstance(inputs, str) or isinstance(inputs, pd.DataFrame), "Inputs should be str_path or df"
        if isinstance(inputs, str):
            df = pd.read_csv(inputs)
        else:
            df = inputs

        try:
            inp_data, out_data = self.__get_inp_out_data(df)

            # get current agent position
            self._pos_agent, self._agent_heading = self.__get_current_agent_state(inp_data)

            # process input
            processed_input = self.__process_input(inp_data)
            # process output
            processed_output = None
            if not is_inference:
                processed_output = self.__process_output(out_data)
            return processed_input, processed_output

        except FailedProcessing:
            return None, None


# --- utility function for DataLoader ---
def collate_fn(
        data: List[Tuple]
):
    """
    :param data: List of (inp, out) data
    :return: stack into batch
    """
    inp_batch, out_batch = process_batch_data(data)
    return inp_batch, out_batch


def process_batch_data(
        batch_data: List[Tuple]
):
    """
    Function to stack data into batch dimension
    :param batch_data: batch data in list
    :return:
        Tuple[Dict, torch.Tensor]
            - Data input stacked in batch dimension
                + traffic_light: (batch, dim)
                + map: (batch, num_waypoints, dim)
                + agent: (batch, history_steps, dim)
                + others: (batch, num_others, history_steps, dim)
            - Data output stacked in batch dimension
                (batch, future_steps, 2)
    """
    # clone data
    batch_data = deepcopy(batch_data)
    # input
    light = list()
    map = list()
    agent = list()
    others = list()
    pad_objects(batch_data)  # padding zeros for objects
    # output
    gt = list()

    # append data
    for inp, out in batch_data:
        light.append(inp["traffic_light"])
        map.append(inp["map"])
        agent.append(inp["agent"])
        others.append(inp["others"])
        gt.append(out)

    # stack batch data
    x = {
        "traffic_light": torch.stack(light, dim=0),
        "map": torch.stack(map, dim=0),
        "agent": torch.stack(agent, dim=0),
        "others": torch.stack(others, dim=0)
    }
    y = torch.stack(gt, dim=0)

    return x, y
