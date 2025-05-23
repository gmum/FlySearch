from PIL import Image

from conversation import BaseConversationFactory, Role, GPTFactory
from rl.agents.semantic_units import BaseSemanticSubunit


class ActionSpaceSpecialistFailure(Exception):
    pass


class ActionSpaceSpecialist(BaseSemanticSubunit):
    def __init__(self, conversation_factory: BaseConversationFactory):
        self.conversation_factory = conversation_factory

    def process_information(self, information: dict) -> dict:
        conversation = self.conversation_factory.get_conversation()
        conversation.begin_transaction(Role.USER)
        conversation.add_text_message(
            "You are an action space specialist. That is, you will receive a snapshot of the current state of our agentic application. The state is represented as pairs of keys and values. You need to make some sense out of it -- that is, you need to understand the task, the current state of this task and formulate sample actions that you deem sensible. Make sure your actions make sense, but you don't need to think whether they are the _best_. The other component of the system is responsible for that. Your actions need to have your justifications, so that the further component can understand them.")

        image_counter = 0

        for key, value in information.items():
            if isinstance(value, str):
                conversation.add_text_message(f"{key}: {value}")
            elif isinstance(value, list):
                conversation.add_text_message(f"List of consecutive values of {key}: {value}")
            elif isinstance(value, Image.Image):
                conversation.add_text_message(f"{key}: Image {image_counter}:")
                conversation.add_image_message(value)
                image_counter += 1
            else:
                raise ActionSpaceSpecialistFailure(
                    f"Unsupported type {type(value)} for key {key}. Supported types are str and list."
                )

        conversation.commit_transaction(send_to_vlm=True)
        _, response = conversation.get_latest_message()
        information["actions"] = response

        return information

    def get_actions(self, information: dict) -> str:
        """
        Get the actions from the information.
        :param information: Information to be processed.
        :return: Actions from the information.
        """
        information = self.process_information(information)
        return information["actions"]


def main():
    conversation_factory = GPTFactory()
    action_space_specialist = ActionSpaceSpecialist(conversation_factory=conversation_factory)
    image = Image.open("/home/dominik/MyStuff/active-visual-gpt/data/burger.png")

    information = {
        "image": image,
        "target": "find something funny",
        "previous_actions": ["humans are funny", "this table is kinda nice"],
        "previous_rewards": [-10, -100],
    }

    actions = action_space_specialist.get_actions(information)
    print(actions)


if __name__ == "__main__":
    main()
