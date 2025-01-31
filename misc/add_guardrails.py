import torchvision
import numpy as np
import cv2
import os

from PIL import Image, ImageDraw, ImageFont


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

    font = ImageFont.truetype("/usr/share/fonts/google-noto/NotoSerif-Bold.ttf",
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


def carthesian(x, y):
    for el_x in x:
        for el_y in y:
            yield el_x, el_y


# It doesn't render dots on the edges of the image.
def dot_matrix_two_dimensional_unreal(img: np.ndarray, w_dots, h_dots, pixel_per_unit=0.42, drone_height=100):
    def get_opposite_color(pixel_color):
        if pixel_color[0] + pixel_color[1] + pixel_color[2] >= 255 * 3 / 2:
            opposite_color = (0, 0, 0)
        else:
            opposite_color = (255, 255, 255)

        return opposite_color

    img = from_opencv_to_pil(img)
    width, height = img.size

    assert width == height

    # FIXME: This code ignores the pixel_per_unit parameter and assumes camera FOV=90 degrees
    # pixel_per_unit = (drone_height / 100) * pixel_per_unit
    pixel_per_unit = (2 * drone_height) / width

    # Unit -> unit used inside of Unreal Engine
    # Pixel -> pixel used in the image
    # Cell -> cell in the grid

    if img.mode != 'RGB':
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img, 'RGB')

    font_location = os.getenv("FONT_LOCATION")

    if font_location is None:
        raise ValueError(
            "Please set the FONT_LOCATION environment variable to the path of the font file, so that grid labels can be annotated.")

    font = ImageFont.truetype(font_location, width // 40)  # Adjust font size if needed; default == width // 40

    pixels_per_cell_w = width / w_dots
    pixels_per_cell_h = height / h_dots

    w_center = w_dots / 2
    h_center = h_dots / 2

    for x, y in carthesian(range(1, w_dots), range(1, h_dots)):
        x_diff = x - w_center
        y_diff = h_center - y

        x_diff_px = x_diff * pixels_per_cell_h
        y_diff_px = y_diff * pixels_per_cell_w

        x_diff_unit = x_diff_px * pixel_per_unit
        y_diff_unit = y_diff_px * pixel_per_unit

        x_diff_unit = int(round(x_diff_unit))
        y_diff_unit = int(round(y_diff_unit))

        x_px = x * pixels_per_cell_h
        y_px = y * pixels_per_cell_w

        pixel_color = img.getpixel((x_px, y_px))
        opposite_color = get_opposite_color(pixel_color)

        circle_radius = width // 240
        draw.ellipse([(x_px - circle_radius, y_px - circle_radius), (x_px + circle_radius, y_px + circle_radius)],
                     fill=opposite_color)

        text_x_px, text_y_px = x_px + 3, y_px

        draw.text((text_x_px, text_y_px), f"({x_diff_unit}, {y_diff_unit})", fill=opposite_color, font=font)

    return from_pil_to_opencv(img)


def from_pil_to_opencv(image):
    return np.array(image)[:, :, ::-1].copy()


def from_opencv_to_pil(image):
    return Image.fromarray(image[:, :, ::-1].copy())


def main():
    image = cv2.imread("../data/sample_images/city2.png")

    image1 = dot_matrix_two_dimensional_unreal(image, 6, 6)

    cv2.imshow("Image1", image1)
    cv2.waitKey(0)

    image2 = dot_matrix_two_dimensional_unreal(image, 6, 6, drone_height=50)

    cv2.imshow("Image2", image2)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
