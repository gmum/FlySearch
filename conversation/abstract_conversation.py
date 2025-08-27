import typing

from enum import Enum
from PIL import Image


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Conversation:

    # Upon calling the method, user signals that he wants to send a message (containing text and images)
    # Cannot be called before commit_transaction() or after rollback_transaction() after begin_transaction() is called
    def begin_transaction(self, role: Role):
        pass

    # Adds text message to be sent (later)
    def add_text_message(self, text: str):
        pass

    # Adds image message to be sent (later)
    def add_image_message(self, image: Image.Image):
        pass

    # Sends all messages added since begin_transaction() was called if send_to_vlm is True
    # Otherwise, messages are only added to the conversation
    def commit_transaction(self, send_to_vlm: bool):
        pass

    # Messages added since begin_transaction() was called are discarded
    def rollback_transaction(self):
        pass

    def get_conversation(self, save_urls=True) -> typing.List[typing.Tuple[Role, str]]:
        pass

    def get_latest_message(self) -> typing.Tuple[Role, str]:
        pass
