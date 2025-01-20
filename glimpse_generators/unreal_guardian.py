import subprocess
import pathlib

from time import sleep


class UnrealGuardian:
    def __init__(self, unreal_binary_path: pathlib.Path):
        self.process = None
        self.unreal_binary_path = unreal_binary_path
        self._start_unreal()

    def _start_unreal(self):
        self.process = subprocess.Popen(
            [str(self.unreal_binary_path), "-RenderOffscreen"])

        sleep(15)

    def reset(self):
        self.process.terminate()
        print("Unreal Guardian: Terminated Unreal process.")
        self._start_unreal()

    @property
    def is_alive(self):
        return self.process.poll() is None


def main():
    unreal_guardian = UnrealGuardian(
        pathlib.Path(
            "/home/dominik/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample")
    )


if __name__ == "__main__":
    main()
