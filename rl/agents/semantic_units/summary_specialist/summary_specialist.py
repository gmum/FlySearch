from typing import Dict
from PIL import Image

from conversation import BaseConversationFactory, Role, GPTFactory
from rl.agents.semantic_units import BaseSemanticSubunit


class SummaryFailure(Exception):
    pass


class SummarySpecialist(BaseSemanticSubunit):
    def __init__(self, conversation_factory: BaseConversationFactory):
        self.conversation_factory = conversation_factory

    def process_information(self, information: Dict) -> Dict:
        conversation = self.conversation_factory.get_conversation()

        conversation.begin_transaction(Role.USER)
        conversation.add_text_message(
            "You are a summary specialist. You need to summarize what has happened to the agent executing an arbitrary task. You will receive information about the task and the previous agent's actions. Summarise them in a concise, yet extremely informative way.  You may need to fill in the gaps. Some of the data labels may be vague, so you need to use your common sense. Put emphasis on things you believe were stupid.")

        image_counter = 0

        for key, value in information.items():
            if isinstance(value, str):
                conversation.add_text_message(f"{key}: {value}")
            elif isinstance(value, Image.Image):
                conversation.add_text_message(f"{key}: Image {image_counter}:")
                conversation.add_image_message(value)
                image_counter += 1
            elif isinstance(value, list):
                conversation.add_text_message(f"List of consecutive values of {key}: {value}")
            else:
                raise SummaryFailure(
                    f"Unsupported type {type(value)} for key {key}. Supported types are str, Image.Image, and list."
                )

        conversation.commit_transaction(send_to_vlm=True)
        _, response = conversation.get_latest_message()

        information["summary"] = response
        return information

    def get_summary(self, information: Dict) -> str:
        """
        Get the summary of the information.
        :param information: Information to be summarized.
        :return: Summary of the information.
        """
        information = self.process_information(information)
        return information["summary"]


def main():
    conversation_factory = GPTFactory()
    summary_specialist = SummarySpecialist(conversation_factory=conversation_factory)
    image = Image.open("/home/dominik/MyStuff/active-visual-gpt/data/burger.png")

    information = {
        "image": image,
        "target": "something funny",
        "previous_actions": [
            "The door in the background.",
            "Maybe the plane on the left?",
            "The burger that is not a burger.",
        ]
    }

    summary = summary_specialist.get_summary(information)
    print(summary)


if __name__ == "__main__":
    main()
