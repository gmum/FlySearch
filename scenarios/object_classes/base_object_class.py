import re

from random import Random

from typing import List
from unrealcv import Client

from glimpse_generators import UnrealException


class BaseObjectClass:
    def __init__(self, spawnable_object_ids: List[str], client: Client):
        self.spawnable_object_ids = spawnable_object_ids
        self.client = client
        self.visible = True

    def _hide_object(self, object_id: str):
        id_number = int(re.findall(r'\d+', object_id)[-1])

        print(f"BaseObjectClass _hide_object, object id_number of {object_id} is {id_number}")

        x = id_number // 1000
        y = id_number // 1000
        z = 0

        self.client.request(f"vset /object/{object_id}/location {x} {y} {z}")

        # Fires need also hiding, as it extinguishes them. Fire may burn even after being moved, which is why we need this.
        if 'Niagara' in object_id or object_id in ["BP_Configuration_13_C_UAID_08BFB8191750E53702_1640163914",
                                 "BP_Configuration_12_C_UAID_08BFB8191750E53702_1361488908",
                                 "SkeletalMeshActor_UAID_08BFB8191750E43702_1589650724"]:
            self.client.request(f"vset /object/{object_id}/hide")

    def hide_all_objects(self):
        if not self.visible:
            return

        try:
            self.visible = False

            for object_id in self.spawnable_object_ids:
                self._hide_object(object_id)
        except UnrealException:
            # We are ignoring this exception because if it is raised it (most likely) means that the object we wanted to hide does not exist
            # Which is expected, as we are hiding all objects from all environments. There won't be objects from city in forest and vice versa.
            pass

    # Returns object id of the object moved
    def move_and_show(self, x: float, y: float, z: float, seed: int) -> str:
        self.visible = True

        rng = Random()
        rng.seed(seed)
        object_id = rng.choice(self.spawnable_object_ids)

        self.client.request(f"vset /object/{object_id}/location {x} {y} {z}")
        self.client.request(f"vset /object/{object_id}/show")

        return object_id

    def rotate_object(self, object_id: str, p: float, q: float, r: float):
        self.client.request(f"vset /object/{object_id}/rotation {p} {q} {r}")
