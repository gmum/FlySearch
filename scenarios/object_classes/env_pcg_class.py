class EnvPCGClass:
    def __init__(self, pcg_id: str, client):
        self.pcg_id = pcg_id
        self.client = client

    def _wait_for_pcg(self):
        ready = self.client.request(f'vbp {self.pcg_id} IsPCGReady')

        while ready == "false":
            ready = self.client.request(f'vbp {self.pcg_id} IsPCGReady')
            print(f"PCG is not ready while waiting for {self.pcg_id}, sleeping for 0.5 seconds, got:", ready)

    def run_pcg(self, seed: int):
        self.client.request(f"vbp {self.pcg_id} RunPCG {seed}")
        self._wait_for_pcg()


class ForestEnvPCGClass(EnvPCGClass):
    def __init__(self, pcg_id: str, client):
        super().__init__(pcg_id, client)

    def run_pcg(self, seed: int, live_trees_density: float = 0.0, dead_trees_density: float = 0.0, stones: float = 0.0,
                cliffs: float = 0.0):
        self.client.request(
            f"vbp {self.pcg_id} RunPCG {live_trees_density} {dead_trees_density} {stones} {cliffs} {seed}")
        self._wait_for_pcg()
