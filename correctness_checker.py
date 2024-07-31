from openai import OpenAI

from openai_conversation import OpenAIConversation, OpenAITextMessage
from config import OPEN_AI_KEY
from imagenet_classes import id_to_name

client = OpenAI(
    api_key=OPEN_AI_KEY
)


def get_prompt(predicted_class: str, expected_label: int) -> str:
    return f"""
    The model was asked to assign an ImageNet label for a picture. It's answer was {predicted_class}
    The correct label is: {id_to_name[expected_label]}. Did the model answer correctly? Answer ONLY using "YES" or 
    "NO" in capital letters. DO NOT WRITE ANYTHING ELSE.
    
    Note that correct label may contain synonymic names divided by comma. For example, is the model's answer was "tench",
    but the correct answer was "tench, Tinca tinca", the answer should be "YES".

    """


def check_validity_of_answer(predicted_class: str, expected_label: int) -> bool:
    conversation = OpenAIConversation(client)
    prompt = get_prompt(predicted_class, expected_label)
    conversation.send_messages(OpenAITextMessage(prompt))

    response = str(conversation.get_latest_message()).strip().upper()

    if response == "YES":
        return True
    elif response == "NO":
        return False
    else:
        print("Eval model failed; invalid response", response)
        return False


def main():
    print(check_validity_of_answer("goldfish", 1))
    print(check_validity_of_answer("chair", 0))
    print(check_validity_of_answer("fish", 0))


if __name__ == "__main__":
    main()
