import json

from time import sleep

from scenarios.object_classes import BaseObjectClass


class PCGClass(BaseObjectClass):
    def __init__(self, pcg_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pcg_id = pcg_id

    def _hide_object(self, object_id: str):
        super()._hide_object(object_id)
        # Yes, we have to run PCG as the super method will put it somewhere, but you need to regenerate it for it to happen.
        self.client.request(f"vbp {self.pcg_id} RunPCG 0")
        self._wait_for_pcg()

    def _wait_for_pcg(self):
        response = self.client.request(f'vbp {self.pcg_id} IsPCGReady')

        while "false" in response:
            print(f"PCG is not ready while waiting for {self.pcg_id}, sleeping for 0.5 seconds; got:", response)
            sleep(0.5)
            response = self.client.request(f'vbp {self.pcg_id} IsPCGReady')

    def move_and_show(self, x: float, y: float, z: float, seed: int) -> str:
        object_id = super().move_and_show(x, y, z, seed)
        self.client.request(f"vbp {self.pcg_id} RunPCG {seed}")
        self._wait_for_pcg()
        return object_id
