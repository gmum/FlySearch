from typing import Dict

from PIL import Image

from conversation import Role
from rl.agents.semantic_units import BaseSemanticSubunit


class ExecutionSpecialist(BaseSemanticSubunit):
    def __init__(self, conversation_factory):
        self.conversation_factory = conversation_factory

    def process_information(self, information: Dict) -> Dict:
        conversation = self.conversation_factory.get_conversation()
        conversation.begin_transaction(Role.USER)
        conversation.add_text_message(
            "You are an execution specialist. You will receive an instruction our agentic system received for a task; it also includes information about formatting actions. You will also receive the action you need to perform. Your sole responsibility is to translate that action into said action space. That's it.")

        conversation.add_text_message(f"Instruction: {information['instruction']}")
        conversation.add_text_message(f"Action: {information['action']}")

        if 'context' in information:
            assert isinstance(information['context'], dict)

            for key, value in information['context'].items():
                if isinstance(value, str):
                    conversation.add_text_message(f"{key}: {value}")
                elif isinstance(value, list):
                    conversation.add_text_message(f"List of consecutive values of {key}: {value}")
                elif isinstance(value, Image.Image):
                    conversation.add_text_message(f"{key}: Image:")
                    conversation.add_image_message(value)
                else:
                    try:
                        conversation.add_text_message(f"{key}: {str(value)}")
                    except ValueError as e:
                        raise ValueError(
                            f"Unsupported type {type(value)} for key {key}. Supported types are str, Image.Image, and list."
                        )

        conversation.commit_transaction(send_to_vlm=True)

        _, response = conversation.get_latest_message()

        information["formatted_action"] = response
        return information

    def get_formatted_action(self, information: Dict) -> str:
        """
        Get the formatted action from the information.
        :param information: Information to be processed.
        :return: Formatted action from the information.
        """
        information = self.process_information(information)
        return information["formatted_action"]


def main():
    from conversation import GPTFactory
    from prompts import fs1_prompt

    conversation_factory = GPTFactory()
    execution_specialist = ExecutionSpecialist(conversation_factory=conversation_factory)

    information = {
        "instruction": fs1_prompt(10, "red pickup truck", 200),
        "action": "descend by 50 meters",
    }

    formatted_action = execution_specialist.get_formatted_action(information)
    print(formatted_action)


if __name__ == "__main__":
    main()
