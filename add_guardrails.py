import torchvision
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from sympy.physics.quantum.matrixutils import numpy_ndarray


# Taken and repurposed from https://github.com/leixy20/Scaffold/blob/main/image_processor.py
def dot_matrix_two_dimensional(img: np.ndarray, dots_size_w, dots_size_h):
    """
    takes an original image as input, save the processed image to save_path. Each dot is labeled with two-dimensional Cartesian coordinates (x,y). Suitable for single-image tasks.
    control args:
    1. dots_size_w: the number of columns of the dots matrix
    2. dots_size_h: the number of rows of the dots matrix
    """

    img = from_opencv_to_pil(img)

    if img.mode != 'RGB':
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img, 'RGB')

    width, height = img.size
    grid_size_w = dots_size_w
    grid_size_h = dots_size_h
    cell_width = width / grid_size_w
    cell_height = height / grid_size_h

    font = ImageFont.truetype("/usr/share/fonts/truetype/hack/Hack-Bold.ttf",
                              width // 40)  # Adjust font size if needed; default == width // 40

    count = 0
    for j in range(1, grid_size_h):
        for i in range(1, grid_size_w):
            x = int(i * cell_width)
            y = int(j * cell_height)

            pixel_color = img.getpixel((x, y))
            # choose a more contrasting color from black and white
            if pixel_color[0] + pixel_color[1] + pixel_color[2] >= 255 * 3 / 2:
                opposite_color = (0, 0, 0)
            else:
                opposite_color = (255, 255, 255)

            circle_radius = width // 240  # Adjust dot size if needed; default == width // 240
            draw.ellipse([(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)],
                         fill=opposite_color)

            text_x, text_y = x + 3, y
            label_str = f"({i / dots_size_h}, {j / dots_size_w})"
            draw.text((text_x, text_y), label_str, fill=opposite_color, font=font)
            count += 1

    return from_pil_to_opencv(img)


def from_pil_to_opencv(image):
    return np.array(image)[:, :, ::-1].copy()


def from_opencv_to_pil(image):
    return Image.fromarray(image[:, :, ::-1].copy())


def main():
    from vstar_bench_dataset import VstarSubBenchDataset

    ds = VstarSubBenchDataset("/home/dominik/vstar_bench/direct_attributes", transform=from_pil_to_opencv)
    image, _, _, _ = ds[3]

    image = dot_matrix_two_dimensional(image, 5, 5)
    image = from_opencv_to_pil(image)
    image.show()


if __name__ == "__main__":
    main()
