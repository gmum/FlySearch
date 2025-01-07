import random

from typing import List
from unrealcv import Client


class BaseObjectClass:
    def __init__(self, spawnable_object_ids: List[str], client: Client):
        self.spawnable_object_ids = spawnable_object_ids
        self.client = client

    def _hide_object(self, object_id: str):
        self.client.request(f"vset /objects/{object_id}/hide")

    def _show_object(self, object_id: str):
        self.client.request(f"vset /objects/{object_id}/show")

    def hide_all_objects(self):
        for object_id in self.spawnable_object_ids:
            self._hide_object(object_id)

    # Returns object id of the object moved
    def move_and_show(self, x: float, y: float, z: float, seed: int) -> str:
        random.seed(seed)
        object_id = random.choice(self.spawnable_object_ids)

        self.client.request(f"vset /object/{object_id}/location {x} {y} {z}")
        self._show_object(object_id)

        return object_id

    def rotate_object(self, object_id: str, p: float, q: float, r: float):
        self.client.request(f"vset /object/{object_id}/rotation {p} {q} {r}")

    def can_be_seen_from(self, object_id: str, x: float, y: float, z: float) -> bool:
        return "true" in self.client.request(f"vget /camera/{object_id}/cansee  {x} {y} {z}").lower()
