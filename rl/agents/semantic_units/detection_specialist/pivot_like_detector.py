import statistics
from typing import List, Tuple

import numpy as np

from PIL import Image, ImageDraw

from conversation import BaseConversationFactory, Role, GPTFactory
from rl.agents.semantic_units.detection_specialist import BaseDetector


class PivotFailure(Exception):
    pass


class PivotLikeMechanism:
    def __init__(self, image):
        self.image = image.copy()
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
        except statistics.StatisticsError as e:
            print("Error", e)
            x_std = 50
            y_std = 50

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
        self.image_history = []

    def image_to_detections(self, image: Image.Image, target: str) -> List[Tuple[int, int, int, int]]:
        pivot = PivotLikeMechanism(image)

        sampled_point_count = 40

        pivot.sample_new_points(n=sampled_point_count)

        image_history = []

        for _ in range(self.iterations):
            conversation = self.conversation_factory.get_conversation()
            image = pivot.annotate_image()
            image_history.append(image)
            conversation.begin_transaction(Role.USER)
            conversation.add_text_message(
                f"You are a detection specialist who is trying to detect the target. The object of interest is {target}. The image is annotated with lots of dots. Ignore any other numbers that are not on dots. Pick dots that are closest to the target. Just write their numbers and only numbers. Write a few numbers (like 3 or 4). Example: 5 7 8. Do not write anything else. The number is <40.")
            conversation.add_image_message(image)
            conversation.commit_transaction(send_to_vlm=True)

            _, response = conversation.get_latest_message()
            response.replace(",", " ")

            def is_number(s):
                try:
                    int(s)
                    return True
                except ValueError:
                    return False

            index_list = [int(i) for i in response.split() if is_number(i)]
            index_list = [i for i in index_list if i < sampled_point_count]

            if len(index_list) == 0:
                raise PivotFailure("No points selected")

            pivot.filter_points(index_list)
            pivot.sample_from_previous_distribution(n=sampled_point_count)

        image = pivot.annotate_image()
        image_history.append(image)
        self.image_history = image_history

        detections = pivot.points_of_interest
        min_x = min([point[0] for point in detections])
        max_x = max([point[0] for point in detections])
        min_y = min([point[1] for point in detections])
        max_y = max([point[1] for point in detections])

        detections = (min_x, min_y, max_x, max_y)

        return [detections]


def main():
    from matplotlib import pyplot as plt

    # image = Image.open("/home/dominik/MyStuff/active-visual-gpt/data/burger.png")
    image = Image.open("/home/dominik/MyStuff/active-visual-gpt/all_logs/GPT4o-CityNew/283_r0 /0.png")

    detector = PivotLikeDetector(
        conversation_factory=GPTFactory(),
        iterations=3
    )

    bbox = detector.image_to_detections(image, "a plane")[0]
    x_min, y_min, x_max, y_max = bbox

    for img in detector.image_history:
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    draw = ImageDraw.Draw(image)
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=5)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
