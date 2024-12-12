from unrealcv import Client

def main():
    client = Client("localhost", 9000)
    result = client.connect()

    if not result:
        print("UnrealCV server is not running. Something's wrong.")
        return

    client.request("vget /unrealcv/status")
    objects = client.request("vget /objects")

    with open("objects.txt", "w") as f:
        f.write(objects)

    client.disconnect()

if __name__ == "__main__":
    main()