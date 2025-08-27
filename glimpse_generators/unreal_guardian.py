import subprocess
import pathlib

from time import sleep
from typing import Optional


class UnrealGuardian:
    def __init__(self, unreal_binary_path: pathlib.Path):
        self.process: Optional[subprocess.Popen] = None
        self.unreal_binary_path = unreal_binary_path
        self._start_unreal()

    def _start_unreal(self):
        self.process = subprocess.Popen(
            [str(self.unreal_binary_path), "-RenderOffscreen",  "-nosound"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        sleep(120)

    def reset(self):
        self.process.kill()
        print("Unreal Guardian: Terminated Unreal process.")
        self._start_unreal()

    @property
    def is_alive(self):
        return self.process.poll() is None


def main():
    unreal_guardian = UnrealGuardian(
        pathlib.Path(
            "/home/anonymous/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample")
    )


if __name__ == "__main__":
    main()
