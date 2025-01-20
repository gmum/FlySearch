import os

from unrealcv import Client
from glimpse_generators import UnrealGuardian


class UnrealException(Exception):
    pass


class UnrealCVWrapper(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def request(self, *args, **kwargs):
        response = super().request(*args, **kwargs)
        print("Unreal CV Wrapper: request params", args, kwargs, "response", response)

        if "error" in response:
            raise UnrealException(response)

        return response


class UnrealClientWrapper:
    def __init__(self, host, port, unreal_binary_path):
        self.guardian = UnrealGuardian(unreal_binary_path)
        self.host = host
        self.port = port
        self.client = None

        self._initialize_client()

    def _initialize_client(self):
        connection_result = False

        for i in range(11):
            print(f"Trying to connect to UnrealCV server on port {self.port + i}")
            self.client = UnrealCVWrapper((self.host, self.port + i))
            connection_result = self.client.connect()

            if connection_result:
                break

        if not connection_result:
            raise ConnectionError("Failed to connect to UnrealCV server; is it running?")

        self.client.request('vget /unrealcv/status')
        self.client.request('vset /cameras/spawn')
        self.client.request('vset /camera/1/rotation -90 0 0')

    def request(self, *args, **kwargs):
        print("Unreal Client Wrapper: request params", args, kwargs)

        response = None

        while response is None:
            if not self.guardian.is_alive:
                self.guardian.reset()
                self._initialize_client()
            try:
                response = self.client.request(*args, **kwargs)
            except OSError:
                self.guardian.reset()
                self._initialize_client()

        return response

    def disconnect(self):
        self.client.disconnect()
        self.guardian.process.terminate()


def main():
    from time import sleep

    client = UnrealClientWrapper(
        host="localhost",
        port=9000,
        unreal_binary_path="/home/dominik/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample",
    )

    while True:
        client.request("vget /unrealcv/status")
        print("Is alive:", client.guardian.is_alive)
        client.guardian.process.kill()
        sleep(5)
        print("Is alive:", client.guardian.is_alive)
        sleep(10)


if __name__ == "__main__":
    main()
