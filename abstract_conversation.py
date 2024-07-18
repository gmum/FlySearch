import typing


class Message:
    def payload(self) -> dict:
        pass


class Conversation:
    """
    Abstract class meant to represent a conversation between a user and LLM.
    Had to be implemented because the OpenAI vision API is not stateful according to its documentation.
    """

    def send_messages(self, *message: Message) -> None:
        """
        Sends an arbitrary number of messages to the LLM.
        Doesn't return anything, but should mutate the state of the object by adding sent message and model's response.
        """
        pass

    def get_latest_message(self) -> Message:
        pass

    def get_entire_conversation(self) -> typing.List[Message]:
        pass
