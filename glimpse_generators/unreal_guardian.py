import subprocess
import pathlib
import os

from time import sleep
from typing import Optional
from datetime import datetime

class UnrealGuardian:
    def __init__(self, unreal_binary_path: pathlib.Path):
        self.process: Optional[subprocess.Popen] = None
        self.unreal_binary_path = unreal_binary_path
        self.stdout_logfile, self.stderr_logfile = self._create_logfiles()
        self._start_unreal()

    def _create_logfiles(self):
        base_file_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S-FlySearchBinaryLog")
        file_name_stdout = f"{base_file_name}-stdout"
        file_name_stderr = f"{base_file_name}-stderr"
        base_path = pathlib.Path(os.environ["BINARY_LOG_PATH"])
        base_path.mkdir(exist_ok=True, parents=True)

        stdout = open(base_path / file_name_stdout, "w")
        stderr = open(base_path / file_name_stderr, "w")

        return stdout, stderr


    def _start_unreal(self):
        self.process = subprocess.Popen(
            [str(self.unreal_binary_path), "-RenderOffscreen",  "-nosound"],
            stdout=self.stdout_logfile,
            stderr=self.stderr_logfile,
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
