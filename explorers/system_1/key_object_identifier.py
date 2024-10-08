from conversation.abstract_conversation import Conversation, Role

class KeyObjectIdentifier:
    def get_prompt(self, question): return f"""
        You will be presented with a question meant to be answered by another model. The question will be about certain objects in an image. Your task is to identify objects that are mentioned in the question and needed to answer it properly. Separate objects in questions with comma. Don't write anything else.
        
        For example:
        
        Q: Is the dog on the right side of the man with green socks?
        A: dog, man with green socks
        
        Now, let's start.
        
        Question: {question}"""

    def __init__(self, conversation: Conversation):
        self.conversation = conversation

    def identify_key_objects(self, question: str) -> list[str]:
        self.conversation.begin_transaction(Role.USER)
        self.conversation.add_text_message(self.get_prompt(question))
        self.conversation.commit_transaction(send_to_vlm=True)

        response = self.conversation.get_latest_message()[1]

        return response.split(", ")


def main():
    from conversation.openai_conversation import OpenAIConversation
    from openai import OpenAI
    from misc.config import OPEN_AI_KEY

    client = OpenAI(api_key=OPEN_AI_KEY)
    conversation = OpenAIConversation(
        client,
        model_name="gpt-4o",
    )

    key_object_identifier = KeyObjectIdentifier(conversation)


    # Example questions from V*
    print(key_object_identifier.identify_key_objects("What animal is drawn on that red signicade?"))
    print(key_object_identifier.identify_key_objects("Tell me the number of that player who is shooting."))
    print(key_object_identifier.identify_key_objects("What kind of drink can we buy from that vending machine?"))

if __name__ == "__main__":
    main()