import numpy as np
import torch
import torchvision
import cv2

from matplotlib import pyplot as plt

from glimpse_generators.image_glimpse_generator import ImageGlimpseGenerator


class ExplorationVisualizer:
    def __init__(self, glimpse_requests: list[tuple[float, float, float, float]],
                 glimpse_generator: ImageGlimpseGenerator):
        self.glimpse_requests = glimpse_requests
        self.glimpse_generator = glimpse_generator

    def get_glimpse_boxes(self) -> np.ndarray:
        image_with_glimpses = self.glimpse_generator.get_entire_image()

        # Convert image to torch tensor
        image_with_glimpses = torch.tensor(image_with_glimpses).permute(2, 0, 1)

        for coords in self.glimpse_requests:
            x1, y1, x2, y2 = self.glimpse_generator.convert_proportional_coords_to_pixel(*coords)

            image_with_glimpses = torchvision.utils.draw_bounding_boxes(
                image_with_glimpses,
                torch.tensor([[x1, y1, x2, y2]]),
                width=5
            )

        # Convert image back to numpy array
        image_with_glimpses = image_with_glimpses.permute(1, 2, 0).numpy()

        return image_with_glimpses

    def save_glimpse_boxes(self, file_dir: str) -> None:
        glimpse_boxes = self.get_glimpse_boxes()
        cv2.imwrite(file_dir, glimpse_boxes)

    def save_glimpse_list_figure(self, file_dir) -> None:
        fig, axes = plt.subplots(len(self.glimpse_requests), len(self.glimpse_generator.get_glimpse(0, 0, 1, 1)))

        # To avoid dealing with dimensionality, we will iterate over 1 dimension
        glimpse_list = self.get_glimpse_list()

        for glimpse, ax in zip(glimpse_list, axes.flatten()):
            glimpse = glimpse[:, :, ::-1]
            ax.imshow(glimpse)
            ax.axis("off")

        plt.savefig(file_dir)

    def get_glimpse_list(self) -> list[np.ndarray]:
        glimpses = []

        for request in self.glimpse_requests:
            glimpse = self.glimpse_generator.get_glimpse(*request)
            glimpses.extend(glimpse)

        return glimpses

    def save_glimpses_individually(self, file_dir: str) -> None:
        for i, request in enumerate(self.glimpse_requests):
            glimpse = self.glimpse_generator.get_glimpse(*request)
            for j, subglimpse in enumerate(glimpse):
                try:
                    cv2.imwrite(f"{file_dir}_{i}_{j}.jpeg", subglimpse)
                except Exception as e:
                    print("Failed to save glimpse", e)


def main():
    from vstar_bench_dataset import VstarSubBenchDataset
    from cv2_and_numpy import pil_to_opencv
    from image_glimpse_generator import BasicImageGlimpseGenerator

    ds = VstarSubBenchDataset("/home/dominik/vstar_bench/relative_position", transform=pil_to_opencv)
    img, _, _, _ = ds[0]

    glimpse_generator = BasicImageGlimpseGenerator(img)
    visualizer = ExplorationVisualizer([(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)], glimpse_generator)
    boxes = visualizer.get_glimpse_boxes()
    cv2.imshow("Boxes", boxes)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
