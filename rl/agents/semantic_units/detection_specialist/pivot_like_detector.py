import statistics
from typing import List, Tuple

import numpy as np

from PIL import Image, ImageDraw

from conversation import BaseConversationFactory, Role
from rl.agents.semantic_units.detection_specialist import BaseDetector


class PivotLikeMechanism:
    def __init__(self, image):
        self.image = image
        self.points_of_interest = []

    def sample_new_points(self, n=10):
        """
        Sample n points from the image.
        :param n: Number of points to sample.
        :return: List of sampled points.
        """
        import random

        width, height = self.image.size
        for _ in range(n):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            self.points_of_interest.append((x, y))

    def annotate_image(self, annotation_px_size=20):
        """
        Annotate the image with the sampled points.
        :return: Annotated image.
        """
        image = self.image.copy()

        draw = ImageDraw.Draw(image)
        for i, point in enumerate(self.points_of_interest):
            x, y = point
            axis_diff = annotation_px_size // 2
            eps = 3
            axis_diff += eps
            draw.ellipse((x - axis_diff, y - axis_diff, x + axis_diff, y + axis_diff), fill="black")
            draw.text((x, y), str(i), fill="white", font_size=annotation_px_size, anchor="mm")

        return image

    def sample_from_previous_distribution(self, n=10):
        xs = [point[0] for point in self.points_of_interest]
        ys = [point[1] for point in self.points_of_interest]

        x_mean = statistics.mean(xs)
        y_mean = statistics.mean(ys)

        try:
            x_std = statistics.stdev(xs)
            y_std = statistics.stdev(ys)
        except statistics.StatisticsError:
            x_std = 1
            y_std = 1

        new_coordinates = np.random.randn(n, 2) * np.array([x_std, y_std]) + np.array([x_mean, y_mean])
        new_coordinates = np.clip(new_coordinates, 0, np.array(self.image.size) - 1)

        new_coordinates = new_coordinates.astype(int)
        new_coordinates = [tuple(map(int, coord)) for coord in new_coordinates]

        self.points_of_interest = new_coordinates

    def filter_points(self, index_list):
        """
        Retain only the points at the specified indices.
        :param index_list: List of indices to retain.
        :return: None
        """
        self.points_of_interest = [self.points_of_interest[i] for i in index_list]


class PivotLikeDetector(BaseDetector):
    def __init__(self, conversation_factory: BaseConversationFactory, iterations: int = 5):
        super().__init__()
        self.iterations = iterations
        self.conversation_factory = conversation_factory

    def image_to_detections(self, image: Image.Image, target: str) -> List[Tuple[int, int, int, int]]:
        pivot = PivotLikeMechanism(image)
        pivot.sample_new_points(n=20)

        for _ in range(self.iterations):
            conversation = self.conversation_factory.get_conversation()
            image = pivot.annotate_image()
            conversation.begin_transaction(Role.USER)
            conversation.add_text_message(
                f"You are a detection specialist who is trying to detect the target. The object of interest is {target}. The image is annotated with lots of dots. Pick dots that are closest to the target. Just write their numbers and only numbers. Write a few numbers (like 3 or 4). ")
            conversation.add_image_message(image)
            conversation.commit_transaction(send_to_vlm=True)

            _, response = conversation.get_latest_message()

            def is_number(s):
                try:
                    int(s)
                    return True
                except ValueError:
                    return False

            index_list = [int(i) for i in response.split() if is_number(i)]
            pivot.filter_points(index_list)
            pivot.sample_from_previous_distribution(n=20)


def main():
    from matplotlib import pyplot as plt

    image = Image.open("/home/dominik/MyStuff/active-visual-gpt/data/burger.png")

    detector = PivotLikeMechanism(image)

    detector.sample_new_points(20)

    annotated_images = []

    annotated_image = detector.annotate_image()

    annotated_images.append(annotated_image)

    for i in range(10):
        detector.filter_points([0, 1, 2, 3, 4])
        detector.sample_from_previous_distribution(20)
        annotated_image = detector.annotate_image()
        annotated_images.append(annotated_image)

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axs.flat):
        ax.imshow(annotated_images[i])
        ax.axis("off")
        ax.set_title(f"Iteration {i}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
