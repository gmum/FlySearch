from conversation.base_conversation_factory import BaseConversationFactory


def InvalidFactoryFactory(backend):
    class InvalidFactory(BaseConversationFactory):
        def __init__(self, model):
            raise ImportError(
                f"Failed to import {backend} factory beacuse the library is not installed\n"
                f"in order to use {model} please make sure you import it with corresponding\n"
                "uv sync --extra [gemini,anthropic]\n"
            )

        def get_conversation(self):
            f"""
            Raises an ImportError as the {backend} is not installed
            """
            raise ImportError(
                f"Failed to import {backend} factory beacuse the library is not installed\n"
                f"in order to use it please make sure you import it with corresponding\n"
                "uv sync --extra [gemini,anthropic]"
            )

    return InvalidFactory
