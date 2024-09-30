from conversation.abstract_conversation import Conversation, Role
from abstract_response_parser import SimpleResponseParser, AbstractResponseParser
from xml_response_parser import XMLResponseParser
from cv2_and_numpy import opencv_to_pil, pil_to_opencv
from prompt_generation import get_starting_prompt_for_vstar_explorer_xml, \
    get_classification_prompt_for_vstar_explorer_xml, get_classification_prompt_for_vstar_explorer, \
    get_starting_prompt_for_vstar_explorer
from image_glimpse_generator import ImageGlimpseGenerator
from conversation.intern_conversation import get_conversation
from vstar_bench_dataset import VstarSubBenchDataset


class VisualVStarExplorer:
    def __init__(self,
                 conversation: Conversation,
                 question: str,
                 options: list[str],
                 glimpse_generator: ImageGlimpseGenerator,
                 response_parser: AbstractResponseParser,
                 starting_prompt_generator: callable,
                 classification_prompt_generator: callable,
                 number_glimpses: int = 5,
                 ) -> None:
        self.conversation = conversation
        self.response = -1
        self.question = question
        self.options = options
        self.glimpse_generator = glimpse_generator
        self.glimpse_requests = [(0.0, 0.0, 1.0, 1.0)]
        self.number_glimpses = number_glimpses
        self.response_parser = response_parser
        self.failed_coord_request: str | None = None
        self.staring_prompt_generator = starting_prompt_generator
        self.classification_prompt_generator = classification_prompt_generator

    def step(self, x1=0.0, y1=0.0, x2=1.0, y2=1.0, first=False) -> str:
        glimpse = self.glimpse_generator.get_glimpse(x1, y1, x2, y2)

        self.conversation.begin_transaction(Role.USER)

        if first:
            self.conversation.add_text_message(self.staring_prompt_generator(self.number_glimpses))
            self.conversation.add_text_message(
                self.classification_prompt_generator(self.question, self.options))

        for subglimpse in glimpse:
            pil_subglimpse = opencv_to_pil(subglimpse)
            self.conversation.add_image_message(pil_subglimpse)

        self.conversation.commit_transaction(send_to_vlm=True)

        unfiltered = str(self.conversation.get_latest_message()[1])

        return unfiltered

    def answer(self):
        response = self.step(first=True)

        # We are not subtracting one from self.number_glimpses because one step can be used for classification
        # If no classification is given, the model will respond "-1" and that won't be treated as a correct answer
        for _ in range(self.number_glimpses):
            if self.response_parser.is_answer(response):
                self.response = self.response_parser.get_answer(response)
                return
            elif self.response_parser.is_coordinate_request(response):
                coords = self.response_parser.get_coordinates(response)
                self.glimpse_requests.append(coords)
                try:
                    response = self.step(*coords)
                except Exception as e:
                    print("Invalid coordinates", e, response)
                    self.failed_coord_request = response
                    break
            else:
                self.failed_coord_request = response
                break

    def get_glimpse_requests(self) -> list[tuple[float, float, float, float]]:
        return self.glimpse_requests

    def get_response(self) -> int | str:
        return self.response

    def get_failed_coord_request(self) -> str | None:
        return self.failed_coord_request


def main():
    from openai_conversation import OpenAIConversation
    from image_glimpse_generator import GridImageGlimpseGenerator

    # client = OpenAI(api_key=OPEN_AI_KEY)
    # conversation = OpenAIConversation(
    #     client,
    #    model_name="gpt-4o",
    # )

    conversation = get_conversation()

    # image = cv2.imread("sample_images/burger.jpeg")

    ds = VstarSubBenchDataset("/home/dominik/vstar_bench/direct_attributes", transform=pil_to_opencv)
    image, question, options, answer = ds[1]

    print("Q:", question)
    print("O:", options)
    print("A:", answer)

    glimpse_generator = GridImageGlimpseGenerator(image, 5)
    response_parser = XMLResponseParser()

    explorer = VisualVStarExplorer(
        conversation,
        question,
        options,
        glimpse_generator,
        response_parser=response_parser,
        starting_prompt_generator=get_starting_prompt_for_vstar_explorer_xml,
        classification_prompt_generator=get_classification_prompt_for_vstar_explorer_xml,
    )

    explorer.answer()

    print("Answer", explorer.get_response())

    from visualization import ExplorationVisualizer
    visualizer = ExplorationVisualizer(explorer.get_glimpse_requests(), explorer.glimpse_generator)

    # visualizer.save_glimpse_list_figure("debug/glimpse_list.jpeg")
    visualizer.save_glimpses_individually("debug/glimpse")
    visualizer.save_glimpse_boxes("debug/glimpse_boxes.jpeg")

    # print(conversation.get_conversation())
    print(explorer.get_glimpse_requests())
    # print(conversation)


if __name__ == "__main__":
    main()
