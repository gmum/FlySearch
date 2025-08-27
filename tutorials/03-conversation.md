# Conversation

## `AbstractConversation` and children

In this repository, to keep context of conversations with VLMs we use a simple interface, defined in the `AbstractConversation` class in `conversation/abstract_conversation.py` file. 

The interface is pretty straightforward, so we believe it to be a good idea to just show it in this tutorial:

```python 
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
```

You can always implement this interface yourself, and it _should_ generally work, but we mostly use a child class of `AbstractConversation`, i.e. `OpenAIConversation`:
- We use `OpenAIConversation` to deal with OpenAI's models AND Google's models AND vLLM models via their respective compatibility modes (Gemini 2.5 Pro doesn't seem to be working good in compatibility mode, so if you're experiencing trouble with it, you're not alone).
- We use `AnthropicConversation` to deal with Anthropic's models; AnthropicConversation is a child class of `OpenAIConversation`, although we do a few dirty hacks to change as little code as possible.

While interface should give an idea how this all works, let's have a look at `OpenAIConversation` in action:

```python 
    # Assuming we've loaded dotenv and have proper imports    
    client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])
    conversation = OpenAIConversation(
        client=client,
        model_name="gpt-4o"
    )

    image = Image.open("tutorials/images/plot_awareness_chart.png")

    conversation.begin_transaction(Role.USER)

    conversation.add_text_message("Describe this chart.")
    conversation.add_image_message(image)
    conversation.commit_transaction(send_to_vlm=False)

    conversation.begin_transaction(Role.ASSISTANT)
    conversation.add_text_message("This is not a chart, this is a picture of a burger!")
    conversation.commit_transaction(send_to_vlm=False)

    conversation.begin_transaction(Role.USER)
    conversation.add_text_message("Are you sure?")
    conversation.commit_transaction(send_to_vlm=True)

    print(conversation.get_latest_message())
    # (<Role.ASSISTANT: 'assistant'>, 'I apologize for the oversight. Let\'s take a closer look at the chart: [...])

```

This abstraction is rather flexible and allows you to add custom messages to VLM's context and make them look like they were authored by it. We don't make use of that feature in FlySearch, but this _could_ be a potential avenue of improving model's performance (if used smartly).

## Factories 

Because during testing we need to keep separate conversations per trajectory, we also have a defined base class for conversation factories in form of `BaseConversationFactory` (`conversation/base_conversation_factory.py`). Here is the interface:

```python 
class BaseConversationFactory:
    def get_conversation(self):
        raise NotImplementedError()
```
There are a few implementations, namely:
- `AnthropicFactory`
- `GPTFactory`
- `VLLMFactory`
- `GeminiFactory`

Let's have a look of implementation of `GPTFactory`:

```python
import os

from openai import OpenAI, _types

from conversation.base_conversation_factory import BaseConversationFactory
from conversation.openai_conversation import OpenAIConversation


class GPTFactory(BaseConversationFactory):
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])
        self.model_name = model_name.removeprefix("oai-")

    def get_conversation(self):
        return OpenAIConversation(
            self.client,
            model_name=self.model_name,
            max_tokens=_types.NotGiven(), # We have to do this because otherwise GPT-5 would stop working. 4o works with default arguments for this class, but while making this compatible with GPT-5 I've decided to stop passing these arguments altogether as they don't break the 4o.
            temperature=_types.NotGiven()
        )
```
That's it. We can also look at how Gemini factory is implemented:

```python
import os

from openai import OpenAI, _types

from conversation import BaseConversationFactory
from conversation.openai_conversation import OpenAIConversation


class GeminiFactory(BaseConversationFactory):
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=os.environ["GEMINI_AI_KEY"], base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
                             max_retries=100)

        self.model_name = model_name

    def get_conversation(self):
        return OpenAIConversation(
            self.client,
            model_name=self.model_name,
            max_tokens=_types.NotGiven()
        ) # Without this, Gemini 2.5 Flash will give out Nones. Probably due to reasoning tokens. TODO: Pro still fails (sometimes (!)) for some reason.
```

Initialising conversation from a factory is straightforward:

```python
factory = GPTFactory("oai-gpt-4o")
conversation = factory.get_conversation()
```

## How to contribute

What if you've made your own implementation of conversation and its factory for some family of models that is not supported by FlySearch? Well, you should now go to `arg_resolvers/conversation_factory_resolver` which governs resolving user arguments and modify this function here:

```python
    def get_conversation_factory(self, args: Namespace):
        if args.model.startswith('oai-'):
            return GPTFactory(args.model)
        elif args.model.startswith('gemini-'):
            return GeminiFactory(args.model)
        elif args.model.startswith('anthropic-'):
            return AnthropicFactory(args.model)
        elif args.model.startswith("mynewmodel-"): # ADD THIS
            return MyNewFactory(args.model)
        else:
            return VLLMFactory(args.model)
```

When you will be prefixing the `model` argument with `mynewmodel`, your new custom conversation factory will be used.