import numpy as np

from glimpse_generators.image_glimpse_generator import ImageGlimpseGenerator, BasicImageGlimpseGenerator, \
    GridImageGlimpseGenerator
from misc.add_guardrails import dot_matrix_two_dimensional


class TestImageGlimpseGenerator:
    def test_conversion_of_coordinates_works_properly(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        generator = ImageGlimpseGenerator(image)

        x1, y1, x2, y2 = generator.convert_proportional_coords_to_pixel(0.0, 0.0, 1.0, 1.0)

        assert x1 == 0
        assert y1 == 0
        assert x2 == 2
        assert y2 == 2

        x1, y1, x2, y2 = generator.convert_proportional_coords_to_pixel(0.0, 0.0, 0.5, 0.5)

        assert x1 == 0
        assert y1 == 0
        assert x2 == 1
        assert y2 == 1

    def test_get_image_returns_image(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        generator = ImageGlimpseGenerator(image)

        assert np.array_equal(generator.get_entire_image(), image)


class TestBasicImageGlimpseGenerator:
    def test_glimpse_contains_only_one_image(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        generator = BasicImageGlimpseGenerator(image)
        glimpse = generator.get_glimpse(0.0, 0.0, 0.3, 0.8)
        assert len(glimpse) == 1

    def test_first_coordinate_is_vertical(self):
        image_saliency = [
            [1, 2],
            [3, 4]
        ]

        image = np.array([image_saliency, image_saliency, image_saliency], dtype=np.uint8)
        image = np.transpose(image, (1, 2, 0))

        generator = BasicImageGlimpseGenerator(image)

        upper_left = generator.get_glimpse(0.0, 0.0, 0.5, 0.5)[0]
        upper_right = generator.get_glimpse(0.5, 0.0, 1.0, 0.5)[0]
        lower_left = generator.get_glimpse(0.0, 0.5, 0.5, 1.0)[0]
        lower_right = generator.get_glimpse(0.5, 0.5, 1.0, 1.0)[0]

        assert np.array_equal(upper_left, np.array([[[1, 1, 1]]]))
        assert np.array_equal(upper_right, np.array([[[2, 2, 2]]]))
        assert np.array_equal(lower_left, np.array([[[3, 3, 3]]]))
        assert np.array_equal(lower_right, np.array([[[4, 4, 4]]]))

    def test_get_glimpse_returns_raw_glimpse_in_array(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        generator = BasicImageGlimpseGenerator(image)

        raw_glimpse = generator.get_raw_glimpse(0.1, 0.2, 0.3, 0.8)
        glimpse = generator.get_glimpse(0.1, 0.2, 0.3, 0.8)
        glimpse = generator.get_glimpse(0.1, 0.2, 0.3, 0.8)

        assert np.array_equal(raw_glimpse, glimpse[0])

    def test_full_glimpse_returns_image(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        generator = BasicImageGlimpseGenerator(image)

        glimpse = generator.get_glimpse(0.0, 0.0, 1.0, 1.0)

        assert np.array_equal(glimpse[0], image)


class TestGridImageGlimpseGenerator:
    def test_glimpse_contains_only_one_image(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        generator = GridImageGlimpseGenerator(image, 5)
        glimpse = generator.get_glimpse(0.1, 0.4, 0.3, 0.8)
        assert len(glimpse) == 1

    def test_get_glimpse_returns_raw_glimpse_in_array(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        generator = GridImageGlimpseGenerator(image, 5)

        raw_glimpse = generator.get_raw_glimpse(0.1, 0.2, 0.3, 0.8)
        glimpse = generator.get_glimpse(0.1, 0.2, 0.3, 0.8)

        assert np.array_equal(raw_glimpse, glimpse[0])

    def test_full_glimpse_does_not_return_image(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        generator = GridImageGlimpseGenerator(image, 5)

        glimpse = generator.get_glimpse(0.0, 0.0, 1.0, 1.0)

        assert not np.array_equal(glimpse[0], image)

    def test_full_glimpse_is_full_gridline(self):
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        generator = GridImageGlimpseGenerator(image, 5)
        glimpse = generator.get_glimpse(0.0, 0.0, 1.0, 1.0)
        target = dot_matrix_two_dimensional(image, 5, 5)

        assert np.array_equal(glimpse[0], target)
