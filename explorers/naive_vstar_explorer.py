from explorers.vstar_owl_explorer import VstarOwlExplorer
from misc.cv2_and_numpy import pil_to_opencv, opencv_to_pil

class NaiveVstarExplorer(VstarOwlExplorer):
    def __init__(self, image, conversation, question, options):
        super().__init__(image, conversation, question, options)

    def identify_objects(self) -> list[str]:
        return []

def main():
    from datasets.vstar_bench_dataset import VstarSubBenchDataset
    from conversation.openai_conversation import OpenAIConversation
    from openai import OpenAI
    from misc.config import OPEN_AI_KEY

    client = OpenAI(api_key=OPEN_AI_KEY)
    conversation = OpenAIConversation(
        client,
        model_name="gpt-4o",
    )

    ds = VstarSubBenchDataset("/home/anonymous/vstar_bench/direct_attributes", transform=pil_to_opencv)

    image, question, options, answer = ds[1]

    print(question)
    print(options)
    print(answer)

    explorer = NaiveVstarExplorer(image, conversation, question, options)
    response, _, _, _ = explorer.get_answer()

    print("Response:", response)
    print("Conversation", conversation.get_conversation()[0])

if __name__ == "__main__":
    main()