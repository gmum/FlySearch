from unrealcv import Client

def main():
    client = Client("localhost", 9000)
    result = client.connect()

    if not result:
        print("UnrealCV server is not running. Something's wrong.")
        for i in range(1, 11):
            print(f"Trying to connect to UnrealCV server on port {9000 + i}")
            client = Client("localhost", 9000 + i)
            result = client.connect()

            if result:
                break

        if not result:
            print("Failed to connect to UnrealCV server. Exiting.")
            return

    client.request("vget /unrealcv/status")
    objects = client.request("vget /objects")

    with open("objects.txt", "w") as f:
        f.write(objects)

    client.disconnect()

if __name__ == "__main__":
    main()