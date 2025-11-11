import base64
import io
import logging
from time import sleep

from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError

from conversation.abstract_conversation import Conversation, Role


class GeminiConversation(Conversation):
    def __init__(self, client: genai.Client, model_name: str, seed=42, max_tokens=None, temperature=None, top_p=None, thinking_budget=None):
        self.client = client
        self.model_name = model_name
        self.conversation = []  # This will be populated from chat history
        self.seed = seed
        self.max_tokens = max_tokens  # maps to max_output_tokens in Gemini
        self.temperature = temperature
        self.top_p = top_p
        self.thinking_budget = thinking_budget
        self.logger = logging.getLogger(__name__)

        self.transaction_started = False
        self.transaction_role = None
        self.transaction_conversation = {}

    def begin_transaction(self, role: Role):
        if self.transaction_started:
            raise Exception("Transaction already started")

        self.transaction_started = True
        self.transaction_role = role

        role = "user" if role == Role.USER else "assistant"

        self.transaction_conversation = {
            "role": role,
            "content": []
        }

    def add_text_message(self, text: str):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        if self.transaction_conversation['role'] == 'assistant':
            self.transaction_conversation['content'] = text
        else:
            content = self.transaction_conversation["content"]
            content.append({
                "type": "text",
                "text": text
            })

    def add_image_message(self, image: Image.Image):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        
        content = self.transaction_conversation["content"]

        content.append(
            {
                "type": "image",
                "image": image,
            }
        )

    def _to_gemini_parts(self, message_content):
        parts = []
        if isinstance(message_content, str):
            return [types.Part.from_text(text=message_content)]
        elif isinstance(message_content, list):
            for sub in message_content:
                if sub["type"] == "text":
                    parts.append(types.Part.from_text(text=sub["text"]))
                elif sub["type"] == "image":
                    # Convert PIL Image to bytes
                    img = sub["image"]
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=95)
                    buffer.seek(0)
                    parts.append(types.Part.from_bytes(
                        data=buffer.read(),
                        mime_type='image/jpeg'
                    ))
                elif sub["type"] == "image_url":
                    url = sub["image_url"]["url"]
                    if url.startswith("data:image/"):
                        try:
                            base64_data = url.split(",", 1)[1]
                            image_bytes = base64.b64decode(base64_data)
                            # Determine mime type from data URL
                            mime_type = url.split(";")[0].split(":")[1]
                            parts.append(types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type
                            ))
                        except Exception:
                            parts.append(types.Part.from_text("[image]"))
                    else:
                        parts.append(types.Part.from_text(url))
                else:
                    parts.append(types.Part.from_text("[unsupported content]"))
            return parts
        else:
            return [types.Part.from_text("[unsupported content]")]

    def _get_generation_config(self):
        config_dict = {}
        if self.max_tokens is not None:
            config_dict["max_output_tokens"] = self.max_tokens
        if self.temperature is not None:
            config_dict["temperature"] = self.temperature
        if self.top_p is not None:
            config_dict["top_p"] = self.top_p
        if self.thinking_budget is not None:
            config_dict["thinking_budget"] = self.thinking_budget
        return types.GenerateContentConfig(**config_dict) if config_dict else None

    def _send_message_with_retry(self, conversation_history):
        retries = 3
        delay = 5  # seconds
        for i in range(retries):
            try:
                generation_config = self._get_generation_config()
                # Convert conversation history to contents format
                contents = []
                for msg in conversation_history:
                    contents.append(types.Content(
                        role=msg["role"],
                        parts=msg["parts"]
                    ))
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=generation_config
                )
                return response
            except (APIError, ServerError) as e:
                # Using 429 for rate limiting, but being broad for other transient issues
                if e.code in [429, 500, 503, 504]:
                    self.logger.warning(f"APIError received: {e}. Retrying in {delay} seconds...")
                    sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    self.logger.error(f"Unhandled APIError: {e}")
                    raise e
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                raise e
        raise Exception("Failed to get response after multiple retries")

    def commit_transaction(self, send_to_vlm=False):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        message_to_commit = self.transaction_conversation
        self.conversation.append(message_to_commit)

        self.transaction_conversation = {}
        self.transaction_started = False
        
        role = self.transaction_role
        self.transaction_role = None

        if role == Role.ASSISTANT and send_to_vlm:
            raise Exception("Assistant cannot send messages to VLM")

        if not send_to_vlm:
            return
        
        gemini_history = []
        for msg in self.conversation:
            role = "user" if msg["role"] == "user" else "model"
            parts = self._to_gemini_parts(msg["content"])
            gemini_history.append({
                "role": role,
                "parts": parts
            })

        response = self._send_message_with_retry(gemini_history)
        
        response_content = str(response.text)
        
        self.logger.info(f"LLM response: {response_content}")

        # Add the model's response to the history
        response_message = {
            "role": "assistant",
            "content": response_content
        }
        self.conversation.append(response_message)

    def rollback_transaction(self):
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.transaction_conversation = {}

        self.transaction_started = False
        self.transaction_role = None

    def get_conversation(self, save_urls=True):
        def conversation_iterator():
            for message in self.conversation:
                role = Role.USER if message["role"] == "user" else Role.ASSISTANT
                content = message["content"]

                if isinstance(content, str):
                    yield role, content
                elif isinstance(content, list):
                    for submessage in content:
                        if submessage["type"] == "text":
                            yield role, submessage["text"]
                        elif submessage["type"] == "image_url":
                            if save_urls:
                                yield role, submessage["image_url"]
                            else:
                                yield role, "image"
                else:
                    raise Exception("Invalid content type")

        return list(conversation_iterator())

    def get_latest_message(self):
        if len(self.conversation) == 0:
            raise Exception("No messages in conversation")

        return self.get_conversation()[-1]

    def get_entire_conversation(self):
        return self.conversation
