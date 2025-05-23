from typing import Dict

from PIL import Image

from conversation import Role
from rl.agents.semantic_units import BaseSemanticSubunit


class DecisionMakingFailure(Exception):
    pass


class DecisionMakingSpecialist(BaseSemanticSubunit):
    def __init__(self, conversation_factory):
        self.conversation_factory = conversation_factory

    def process_information(self, information: Dict) -> Dict:
        conversation = self.conversation_factory.get_conversation()
        conversation.begin_transaction(Role.USER)

        try:
            actions = information["actions"]
        except KeyError:
            raise DecisionMakingFailure("No actions provided in the information.")

        conversation.add_text_message(
            "You are a rational deicision maker at the end of the decision making process, inside of a large agentic application. Your assistant has prepared a set of actions you can take. You need to choose the best one, based on the context provided. Good luck. We will blame your failures precisely on you.")

        conversation.add_text_message(f"Actions: {actions}")
        conversation.commit_transaction(send_to_vlm=True)

        conversation.begin_transaction(Role.USER)
        conversation.add_text_message(
            "Now, let's summarize your desired action in a single sentence. The executor down the line (which is not a smart piece of a LLM) must understand it. Please be concise and clear. Do not justify your decision, just write it. No formatting, no stuff.")

        conversation.commit_transaction(send_to_vlm=True)

        _, response = conversation.get_latest_message()
        information["decision"] = response
        return information

    def get_decision(self, information: Dict) -> str:
        """
        Get the decision from the information.
        :param information: Information to be processed.
        :return: Decision from the information.
        """
        information = self.process_information(information)
        return information["decision"]


def main():
    from conversation import GPTFactory
    from rl.agents.semantic_units.action_space_specialist import ActionSpaceSpecialist

    conversation_factory = GPTFactory()
    action_space_specialist = ActionSpaceSpecialist(conversation_factory=conversation_factory)
    image = Image.open("/home/dominik/MyStuff/active-visual-gpt/data/burger.png")

    information = {
        "image": image,
        "target": "find something funny",
        "previous_actions": ["humans are funny", "this table is kinda nice"],
        "previous_rewards": [-10, -100],
    }

    actions = action_space_specialist.process_information(information)

    decision_making_specialist = DecisionMakingSpecialist(conversation_factory=conversation_factory)

    decision = decision_making_specialist.get_decision(actions)
    print(decision)


if __name__ == "__main__":
    main()
