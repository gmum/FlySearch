import pathlib

from glimpse_generators import UnrealClientWrapper
from glimpse_generators.unreal_glimpse_generator import UnrealGridGlimpseGenerator


class TestUnrealImageGenerator:
    def test_linear(self):
        test_path = pathlib.Path("../all_logs/test_unreal_image_generator/linear_5y")
        test_path.mkdir(exist_ok=True)

        client = UnrealClientWrapper(host="localhost", port=9000,
                                     unreal_binary_path="/home/anonymous/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample"
                                     # unreal_binary_path="/home/anonymous/MyStuff/simulator/CitySample/Binaries/Linux/CitySample"
                                     )

        generator = UnrealGridGlimpseGenerator(splits_w=6, splits_h=6, client=client)

        n = 100
        min_y = 200
        max_y = 300

        start_x = -30
        start_z = 15

        for y in range(min_y, max_y, 10):
            current_point = (start_x, y, start_z)
            image = generator.get_camera_image(current_point)

            image.save(test_path / f"{y}.png")

    def test_linear_grid(self):
        test_path = pathlib.Path("../all_logs/test_unreal_image_generator/linear_grid3")
        test_path.mkdir(exist_ok=True)

        generator = UnrealGridGlimpseGenerator(port=9000, splits_w=6, splits_h=6)

        n = 100
        min_z = 5
        max_z = 120

        start_x = -50
        start_y = -55

        for z in range(min_z, max_z, 5):
            current_point = (start_x, start_y, z)
            image = generator.get_camera_image(current_point)

            image.save(test_path / f"{z}_0.png")

            image = generator.get_camera_image((start_x + 5, start_y, z))
            image.save(test_path / f"{z}_5.png")
