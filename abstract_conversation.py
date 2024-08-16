import typing

from PIL import Image


class Conversation:

    # Upon calling the method, user signals that he wants to send a message (containing text and images)
    # Cannot be called before commit_transaction() or after rollback_transaction() after begin_transaction() is called
    def begin_transaction(self):
        pass

    # Adds text message to be sent (later)
    def add_text_message(self, text: str):
        pass

    # Adds image message to be sent (later)
    def add_image_message(self, image: Image.Image):
        pass

    # Sends all messages added since begin_transaction() was called to the VLM
    def commit_transaction(self):
        pass

    # Messages added since begin_transaction() was called are discarded
    def rollback_transaction(self):
        pass
