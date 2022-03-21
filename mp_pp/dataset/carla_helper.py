import json
import math
import pandas as pd
import numpy as np
from typing import List, Tuple
from dataset.utils import rotate
from copy import deepcopy


class MapHelper:
    """
    This class supports to process map data
    """
    def __init__(self, map_path):
        self._columns = [
            "id",
            "type",
            "x",
            "y",
            "status"
        ]
        self.distance = 50.
        self._get_map(map_path)
        self._get_center_lane_polyline()

    def _get_map(
            self,
            map_path: str,
    ):
        """
        Load data map into self._map
        :param map_path:
        :return:
        """
        df = pd.read_csv(map_path)
        self._map = df
        # # get lane only
        # self._map = self._map.loc[
        #     (self._map["type"] == "l_lane") |
        #     (self._map["type"] == "r_lane")
        #     ]

    def _get_center_lane_polyline(self):
        """
        Get center of each lane polyline
        Result is reserved in self._center_polyline
        :return:
        """
        # deep copy all components in map
        self._center_polyline = self._map.copy(deep=True)

        # group by id and type, and get mean position
        self._center_polyline = self._center_polyline.groupby(
            by=self._columns[:2], as_index=False
        ).mean()

    @staticmethod
    def __estimate_distance(
            row,
            agent_x,
            agent_y
    ):
        """
        Estimate distance of agent to center lane polyline
        Result is reserved in new column "distance"
        :param row: each row in dataframe
        :param agent_x: current position of agent in x-coordinate
        :param agent_y: current position of agent in y-coordinate
        :return:
        """
        row["distance"] = math.sqrt(
            (row["x"] - agent_x) ** 2 +
            (row["y"] - agent_y) ** 2
        )
        return row

    def _get_id_lane_in_range(
            self,
            agent_x: float,
            agent_y: float,
    ) -> List:
        """
        Get id of lane in range distance
        Args:
            agent_x (float): x-coordinate of AGENT
            agent_y (float): y-coordinate of AGENT

        Returns:
            (List[int]): list of AGENT's index
        """
        tmp_center_polyline = self._center_polyline.copy(deep=True)
        tmp_center_polyline = tmp_center_polyline.apply(
            self.__estimate_distance,
            axis=1,  # apply for each row
            agent_x=agent_x,
            agent_y=agent_y
        )
        # get polyline that have distance to agent <= "distance"
        tmp_center_polyline = tmp_center_polyline.loc[
            tmp_center_polyline["distance"] <= self.distance]
        # get ids in list
        ids = tmp_center_polyline[self._columns[0]].to_numpy().tolist()
        return ids

    def _get_lane_by_id(
            self,
            list_ids: List
    ) -> List:
        """
        Get list of lane by lane_ids
        :param list_ids: list index of lane
        :return:
            (List) List of lane by id
        """
        tmp_map = self._map.copy(deep=True)
        tmp_map = tmp_map.loc[tmp_map["id"].isin(list_ids)]

        # group by id, type
        lane_by_id = tmp_map.groupby(by=self._columns[:2])

        # reserve into list container
        lanes = []
        for id_type, frame in lane_by_id:
            # only get numpy of (x, y)
            data = frame[self._columns[2:4]].to_numpy()
            lanes.append(data)
        return lanes

    @staticmethod
    def _get_lane_on_direction(
            ids: List,
            lanes: List[np.ndarray],
            agent_x: float,
            agent_y: float,
            heading: float
    ) -> Tuple[List, List[np.ndarray]]:
        """
        Get lane on the same direction with AGENT
        Args:
            lanes (List[np.ndarray]): list of lanes
            agent_x (float): x-coordinate of AGENT
            agent_y (float): y-coordinate of AGENT
            heading (float): yaw of AGENT

        Returns:
            Tuple[List, List[np.ndarray]]:
                - index of lane
                - lanes on AGENT's direction
        """

        def __dist(p1, p2):
            _x1, _y1 = p1
            _x2, _y2 = p2
            return math.sqrt((_x1 - _x2) ** 2 + (_y1 - _y2) ** 2)

        def __angular(v1, v2):
            _x1, _y1 = v1
            _x2, _y2 = v2
            _norm_v1 = math.sqrt(_x1 ** 2 + _y1 ** 2)
            _norm_v2 = math.sqrt(_x2 ** 2 + _y2 ** 2)
            _a = _x1 * _x2 + _y1 * _y2
            _b = _norm_v1 * _norm_v2
            ratio = round(_a / _b, 5)
            return math.acos(ratio)

        def __initial_adding():
            """
            Initial phase (get the first lane)
            Get the nearest lane line
            with diff angular of vector heading and the first vector lane <= 10 degree
            Returns:
                False: to re-assign _init flag
            """
            _lanes_clone = deepcopy(lanes)
            _ids_clone = deepcopy(ids)

            unit_vector = (1, 0)
            forward_vector = rotate(unit_vector[0], unit_vector[1], heading)

            while len(_lanes_clone) > 0:
                _np_points = np.array(_lanes_clone).reshape((-1, 2))
                _dist_to_agent = (curr_lane[-1] - _np_points) ** 2
                _dist_to_agent = np.sqrt(_dist_to_agent[:, 0] + _dist_to_agent[:, 1])
                _idx_point = np.argmin(_dist_to_agent)
                # get index of lane that containing nearest point
                _i = _idx_point // 10  # 10 point per lane

                # check is legal lane?
                _lane = _lanes_clone[_i]
                v2 = _lane[2] - _lane[0]
                _angular = __angular(forward_vector, v2)
                # if diff in 10 degree
                if _angular <= math.radians(10):
                    # create new lane
                    # by interpolating from agent position
                    # to tail point of the nearest lane
                    _head_point = np.array([agent_x, agent_y])
                    _tail_point = _lane[-1]

                    # skip if distance of a lane
                    # is too short (2m)
                    if np.linalg.norm(_tail_point - _head_point) >= 2:
                        _x = np.linspace(_head_point[0], _tail_point[0], 10)
                        _y = np.linspace(_head_point[1], _tail_point[1], 10)
                        _interp_lane = np.stack([_x, _y], axis=1)

                        queue.append(_interp_lane)
                        res_ids.append(_ids_clone[_i])
                        res_lanes.append(_interp_lane)
                        return

                _ids_clone.pop(_i)
                _lanes_clone.pop(_i)

            return False

        def __adding():
            """
            Chaining phase (get next lane)
            Get the lane in range and fit the angular with previous lane
            Returns:

            """
            for _i, (_id, _lane) in enumerate(zip(ids, lanes)):
                _distance = __dist(curr_lane[-1], _lane[0])
                # get lane in range
                if _distance >= 10:
                    continue
                v1 = curr_lane[-1] - curr_lane[-3]
                v2 = _lane[2] - _lane[0]
                if __angular(v1, v2) <= math.radians(10):
                    queue.append(_lane)

                    res_ids.append(_id)
                    res_lanes.append(_lane)

                    ids.pop(_i)
                    lanes.pop(_i)

        # --- main flow ---
        res_ids = list()
        res_lanes = list()
        queue = [np.array([[agent_x, agent_y]])]
        _init = True

        while len(queue) > 0:
            # trace queue
            # set the last point of polyline as new curr_pos
            curr_lane = queue.pop(0)

            # if first run
            if _init:
                _init = __initial_adding()
            # get next lane
            else:
                __adding()

        return res_ids, res_lanes

    def get_local_lanes(
            self,
            agent_x: float,
            agent_y: float,
            heading: float
    ) -> Tuple[List, List[np.ndarray]]:
        """
        Get local lane on the way of AGENT, the lane must be:
            - in range distance
            - have the same direction of AGENT
            - not backward (behind) AGENT
        Args:
            agent_x (float): x-coordinate of AGENT
            agent_y (float): y-coordinate of AGENT
            heading (float): yaw of AGENT
            (all values are in world coordinates)

        Returns:
            (List, List[np.ndarray]):
                - index of lanes
                - list of lanes
        """
        # get all lane_id in range distance
        ids = self._get_id_lane_in_range(agent_x, agent_y)
        # get lane polyline
        lanes = self._get_lane_by_id(ids)
        # get lane on AGENT's direction only
        ids, lanes = self._get_lane_on_direction(ids, lanes, agent_x, agent_y, heading)

        return ids, lanes

    def get_status(self, lane_id):
        """
        Get status of corresponding lane_id
        Args:
            lane_id (int): index of lane

        Returns:
            (Dict): status of lane
        """
        status = self._map.loc[self._map["id"] == lane_id, "status"].iloc[0]
        return json.loads(status)
