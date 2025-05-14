import pathlib
import os
import json

from PIL import Image
from typing import Dict, List, Tuple, Optional


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

    @staticmethod
    def _load_comments(path: pathlib.Path) -> List[str]:
        try:
            all_comments = json.load(open(path / 'simple_conversation.json'))
        except FileNotFoundError:
            all_comments = json.load(open(path / 'conversation.json'))
        return [comment for role, comment in all_comments if role == "Role.ASSISTANT" or role == "assistant"]\

    @staticmethod
    def _load_object_bbox(path: pathlib.Path) -> Tuple:
        with open(path / 'object_bbox.txt') as f:
            tuple_str = f.read()
            return tuple([float(coord) for coord in tuple_str.split()])

    @staticmethod
    def _load_username(path: pathlib.Path) -> Optional[str]:
        try:
            with open(path / 'user.txt') as f:
                username = f.read().strip()
                return username
        except FileNotFoundError:
            return None

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.params = self._load_params(path)
        self.coords = self._load_coords(path)
        self.comments = self._load_comments(path)
        self.object_bbox = self._load_object_bbox(path)
        self.username = self._load_username(path)

    def get_params(self):
        return self.params

    def get_coords(self):
        return self.coords

    def get_comments(self):
        return self.comments

    def get_object_bbox(self):
        return self.object_bbox

    def get_username(self):
        return self.username

    def username_recorded(self):
        return self.username is not None

    def get_images(self) -> List[Image]:
        names = [name for name in os.listdir(self.path) if name.endswith(".png")]
        return [Image.open(self.path / name) for name in names]

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
        return self._convert_tuple_str(self.get_params()["drone_rel_coords"])

    @property
    def end_position(self):
        return self.get_coords()[-1]

    @property
    def model_claimed(self):
        try:
            return "found" in self.comments[-1].lower()
        except IndexError:
            return False

    @property
    def real_object_coords(self):
        return Run._convert_tuple_str(self.params["object_coords"])
