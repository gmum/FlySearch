import pathlib
import os
import json

from typing import Dict, List, Tuple

from requests.cookies import get_cookie_header
from sympy.codegen.cnodes import static


class Run:
    @staticmethod
    def _convert_tuple_str(tuple_str: str) -> Tuple:
        tuple_str = tuple_str.replace("(", "").replace(")", "").strip().split(",")
        return tuple([float(coord) for coord in tuple_str])



    @staticmethod
    def _load_params(path: pathlib.Path) -> Dict:
        with open(path / 'scenario_params.json') as f:
            params = json.load(f)
            return params

    @staticmethod
    def _load_coords(path: pathlib.Path) -> List[Tuple]:
        coord_list = []
        coord_files = [name for name in os.listdir(path) if name.endswith("coords.txt")]

        def number_before_dash(name):
            try:
                int(name.split('_')[0])
                return True
            except ValueError:
                return False

        coord_files = [name for name in coord_files if number_before_dash(name)]

        def cmp(name):
            return int(name.split('_')[0])

        coord_files.sort(key=cmp)

        for name in coord_files:
            with open(path / name) as f:
                tuple_str = f.read()
                coord_list.append(Run._convert_tuple_str(tuple_str))

        return coord_list

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.params = self._load_params(path)
        self.coords = self._load_coords(path)

    def get_params(self):
        return self.params

    def get_coords(self):
        return self.coords

    @property
    def forest_level(self):
        return "regenerate_forest" in self.params and self.params["regenerate_forest"]

    @property
    def seed(self):
        return int(self.params["seed"])

    @property
    def object_type(self):
        return self.params["object_type"].removeprefix("ObjectType.")

    @property
    def start_position(self):
        return self.get_coords()[0]

    @property
    def end_position(self):
        return self.get_coords()[-1]