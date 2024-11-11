from conversation.intern_conversation import InternConversation, get_model_and_stuff


class InternFactory:
    def __init__(self):
        self.model_and_stuff = get_model_and_stuff()

    def get_conversation(self):
        return InternConversation(**self.model_and_stuff)
