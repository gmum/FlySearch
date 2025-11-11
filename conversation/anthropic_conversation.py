import base64
import io
import logging
from dataclasses import dataclass
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image

from conversation.abstract_conversation import Conversation, Role

# Conditional import for optional anthropic dependency
# This allows the file to be imported even if anthropic is not installed
if TYPE_CHECKING:
    from anthropic import Anthropic

# Try to import at runtime, but don't fail if not available
try:
    from anthropic import APIError, InternalServerError, RateLimitError

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    APIError = Exception  # Fallback to base Exception
    RateLimitError = Exception  # Fallback to base Exception
    InternalServerError = Exception  # Fallback to base Exception

# Constants
JPEG_QUALITY = 95
MAX_RETRIES = 10
BASE_RETRY_DELAY_SECONDS = 10
MAX_RETRY_DELAY_SECONDS = 300


@dataclass
class AnthropicResponseAdapter:
    """
    Adapter to provide a consistent interface for Anthropic API responses.
    Captures both content and metadata for monitoring and optimization.
    """
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None
    model: Optional[str] = None

    @classmethod
    def from_anthropic_response(cls, response: Any) -> 'AnthropicResponseAdapter':
        """
        Create adapter from Anthropic API response.
        
        Args:
            response: Raw response from Anthropic API
            
        Returns:
            AnthropicResponseAdapter with content and metadata
        """
        content = str(response.content[0].text)

        # Extract usage metadata if available
        usage = getattr(response, 'usage', None)
        input_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
        output_tokens = getattr(usage, 'output_tokens', 0) if usage else 0

        return cls(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=getattr(response, 'stop_reason', None),
            model=getattr(response, 'model', None)
        )


class AnthropicConversation(Conversation):
    """
    Conversation implementation for Anthropic's Claude API.
    
    This class manages multi-turn conversations with Claude, supporting both
    text and image inputs. It implements a transaction-based message system
    where messages are buffered before being sent to the API.
    """

    def __init__(
            self,
            client: 'Anthropic',
            model_name: str,
            seed: int = 42,
            max_tokens: int = 1000,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            system_prompt: Optional[str] = None
    ) -> None:
        """
        Initialize an Anthropic conversation.
        
        Args:
            client: Anthropic API client instance
            model_name: Name of the Claude model to use (e.g., "claude-3-opus-20240229", 
                       "claude-3-7-sonnet-20250219" for extended thinking)
            seed: Random seed (note: Anthropic API doesn't currently support seed parameter)
            max_tokens: Maximum number of tokens in the response
            temperature: Sampling temperature (0.0 to 1.0). Optional, uses API default if not set.
                        Note: Anthropic API constraint - only one of temperature or top_p can be used.
            top_p: Nucleus sampling parameter (0.0 to 1.0). Optional, uses API default if not set.
                   Note: Anthropic API constraint - only one of temperature or top_p can be used.
            system_prompt: Optional system prompt to set behavior/personality across the conversation
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The 'anthropic' package is required to use AnthropicConversation. "
                "Install it with: pip install anthropic"
            )

        # Anthropic API constraint: cannot specify both temperature and top_p
        if temperature is not None and top_p is not None:
            raise ValueError(
                "Anthropic API does not allow both 'temperature' and 'top_p' to be specified. "
                f"Got temperature={temperature} and top_p={top_p}. "
                "Please use only one or leave both as None to use API defaults."
            )

        self.client = client
        self.model_name = model_name
        self.seed = seed  # Stored but not used (Anthropic doesn't support seed)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(__name__)

        # Conversation state
        self.conversation: List[Dict[str, Any]] = []

        # Transaction state
        self.transaction_started = False
        self.transaction_role: Optional[Role] = None
        self.transaction_conversation: Dict[str, Any] = {}

        # Image counter tracks total images added during current transaction
        self.image_counter = 0
        # Snapshot of image_counter after last committed transaction (for rollback)
        self.post_transaction_image_counter = 0

        # Token usage tracking for context management
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def begin_transaction(self, role: Role) -> None:
        """
        Begin a new message transaction.
        
        Args:
            role: The role of the message (USER or ASSISTANT)
            
        Raises:
            Exception: If a transaction is already in progress
        """
        if self.transaction_started:
            raise Exception("Transaction already started")

        self.transaction_started = True
        self.transaction_role = role

        role_str = "user" if role == Role.USER else "assistant"

        self.transaction_conversation = {
            "role": role_str,
            "content": []
        }

    def add_text_message(self, text: str) -> None:
        """
        Add a text message to the current transaction.
        
        Args:
            text: The text content to add
            
        Raises:
            Exception: If no transaction is in progress
        """
        if not self.transaction_started:
            raise Exception("Transaction not started")

        if self.transaction_conversation['role'] == 'assistant':
            # Assistant messages are simple strings in Anthropic's format
            self.transaction_conversation['content'] = text
        else:
            # User messages are arrays of content blocks
            content = self.transaction_conversation["content"]
            content.append({
                "type": "text",
                "text": text
            })

    def add_image_message(self, image: Image.Image) -> None:
        """
        Add an image message to the current transaction.
        
        Images are converted to JPEG format and base64 encoded.
        A text label is automatically added before each image.
        
        Args:
            image: PIL Image to add to the conversation
            
        Raises:
            Exception: If no transaction is in progress
        """
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.image_counter += 1
        self.add_text_message(f"Image {self.image_counter}:")

        # Convert image to JPEG and base64 encode
        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=JPEG_QUALITY)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        content = self.transaction_conversation["content"]
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": str(base64_image)
            }
        })

    def _call_anthropic_api(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        max_tokens: int, 
        temperature: Optional[float], 
        top_p: Optional[float],
        system_prompt: Optional[str] = None
    ) -> AnthropicResponseAdapter:
        """
        Call the Anthropic API with retry logic and exponential backoff.
        
        Retries on transient errors (rate limits, server overload) with exponential backoff.
        Does not retry on client errors (invalid requests, authentication failures).
        
        Note: Anthropic API only allows one of temperature or top_p to be specified.
        
        Args:
            model: Model name to use
            messages: List of conversation messages
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (optional)
            top_p: Nucleus sampling parameter (optional)
            system_prompt: Optional system prompt for the conversation
            
        Returns:
            AnthropicResponseAdapter wrapping the API response with metadata
            
        Raises:
            RateLimitError: If rate limit exceeded after all retries
            InternalServerError: If server errors persist after all retries
            APIError: For client errors (not retried)
            Exception: For unexpected errors
        """
        for attempt in range(MAX_RETRIES):
            try:
                # Prepare API call parameters
                api_params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    # Note: Anthropic API doesn't currently support seed parameter for reproducibility
                    # See: https://docs.anthropic.com/claude/reference/messages_post
                }

                # Anthropic API constraint: temperature and top_p cannot both be specified
                # Only add the parameter if it's explicitly set
                if temperature is not None:
                    api_params["temperature"] = temperature
                if top_p is not None:
                    api_params["top_p"] = top_p

                # Add system prompt if provided
                if system_prompt:
                    api_params["system"] = system_prompt

                response = self.client.messages.create(**api_params)
                adapter = AnthropicResponseAdapter.from_anthropic_response(response)

                # Track token usage for context management
                self.total_input_tokens += adapter.input_tokens
                self.total_output_tokens += adapter.output_tokens

                self.logger.debug(
                    f"API call successful. Input tokens: {adapter.input_tokens}, "
                    f"Output tokens: {adapter.output_tokens}, "
                    f"Total conversation tokens: {self.total_input_tokens + self.total_output_tokens}"
                )

                return adapter

            except RateLimitError as e:
                if attempt == MAX_RETRIES - 1:
                    self.logger.error(f"Rate limit error after {MAX_RETRIES} attempts")
                    raise

                # Exponential backoff with cap
                delay = min(BASE_RETRY_DELAY_SECONDS * (2 ** attempt), MAX_RETRY_DELAY_SECONDS)
                self.logger.warning(
                    f"Rate limit error (attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Retrying in {delay}s... Error: {e}"
                )
                sleep(delay)
            
            except InternalServerError as e:
                # Retry on server errors (500, overloaded, etc.)
                if attempt == MAX_RETRIES - 1:
                    self.logger.error(f"Server error after {MAX_RETRIES} attempts: {e}")
                    raise

                # Exponential backoff with cap
                delay = min(BASE_RETRY_DELAY_SECONDS * (2 ** attempt), MAX_RETRY_DELAY_SECONDS)
                self.logger.warning(
                    f"Server error (attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Retrying in {delay}s... Error: {e}"
                )
                sleep(delay)

            except APIError as e:
                # Don't retry on general API errors (e.g., invalid request, authentication)
                self.logger.error(f"Anthropic API error (not retrying): {e}")
                raise

            except Exception as e:
                self.logger.error(f"Unexpected error calling Anthropic API: {e}")
                raise

        raise Exception(f"Failed to get response from Anthropic API after {MAX_RETRIES} attempts")

    def commit_transaction(self, send_to_vlm: bool = False) -> None:
        """
        Commit the current transaction and optionally send to the LLM.
        
        Args:
            send_to_vlm: If True, send the message to Claude and get a response
            
        Raises:
            Exception: If no transaction is in progress or if assistant tries to send to LLM
        """
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.conversation.append(self.transaction_conversation)
        self.transaction_conversation = {}

        if self.transaction_role == Role.ASSISTANT and send_to_vlm:
            raise Exception("Assistant cannot send messages to VLM")

        self.transaction_started = False
        self.transaction_role = None

        # Update image counter snapshot after successful commit
        self.post_transaction_image_counter = self.image_counter

        if not send_to_vlm:
            return

        # Send to Anthropic API and get response
        response = self._call_anthropic_api(
            model=self.model_name,
            messages=self.conversation,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            system_prompt=self.system_prompt
        )

        response_content = response.content
        self.logger.info(f"LLM response: {response_content}")
        self.logger.debug(
            f"Response metadata - Stop reason: {response.stop_reason}, "
            f"Model: {response.model}"
        )

        # Add assistant's response to conversation
        self.begin_transaction(Role.ASSISTANT)
        self.add_text_message(response_content)
        self.commit_transaction(send_to_vlm=False)

    def rollback_transaction(self) -> None:
        """
        Rollback the current transaction, discarding all uncommitted messages.
        
        Raises:
            Exception: If no transaction is in progress
        """
        if not self.transaction_started:
            raise Exception("Transaction not started")

        self.transaction_conversation = {}
        self.transaction_started = False
        self.transaction_role = None

        # Restore image counter to last committed state
        self.image_counter = self.post_transaction_image_counter

    def get_conversation(self, save_urls: bool = True) -> List[Tuple[Role, str]]:
        """
        Get the conversation history as a list of (role, content) tuples.
        
        Args:
            save_urls: If True, include base64 image data; if False, use placeholder "image"
            
        Returns:
            List of (Role, str) tuples representing the conversation
            
        Raises:
            Exception: If content type is invalid
        """

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
                        elif submessage["type"] == "image":
                            if save_urls:
                                yield role, submessage["source"]
                            else:
                                yield role, "image"
                        else:
                            raise Exception(f"Invalid content type: {submessage['type']}")
                else:
                    raise Exception(f"Invalid content format: {type(content)}")

        return list(conversation_iterator())

    def get_latest_message(self) -> Tuple[Role, str]:
        """
        Get the most recent message from the conversation.
        
        Returns:
            Tuple of (Role, str) representing the latest message
            
        Raises:
            Exception: If conversation is empty
        """
        if len(self.conversation) == 0:
            raise Exception("No messages in conversation")

        return self.get_conversation()[-1]

    def get_entire_conversation(self) -> List[Dict[str, Any]]:
        """
        Get the raw conversation history.
        
        Returns:
            List of message dictionaries in Anthropic's format
        """
        return self.conversation

    def get_token_usage(self) -> Dict[str, int]:
        """
        Get token usage statistics for the conversation.
        
        Returns:
            Dictionary with input_tokens, output_tokens, and total_tokens
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }

    def estimate_remaining_context(self, context_limit: int = 200000) -> int:
        """
        Estimate remaining context window based on token usage.
        
        Note: This is an approximation. The actual context window may vary
        depending on the model and how tokens are counted (input vs output).
        
        Args:
            context_limit: Maximum context window size (default: 200K for Claude Sonnet 4.5)
            
        Returns:
            Estimated remaining tokens in context window
        """
        total_tokens = self.total_input_tokens + self.total_output_tokens
        remaining = context_limit - total_tokens

        if remaining < 10000:  # Less than 10K tokens remaining
            self.logger.warning(
                f"Context window approaching limit. Used: {total_tokens}/{context_limit} tokens. "
                f"Remaining: {remaining} tokens"
            )

        return max(0, remaining)

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Update the system prompt for future API calls.
        
        Note: This only affects new API calls, not past messages in the conversation.
        
        Args:
            system_prompt: New system prompt to use
        """
        self.system_prompt = system_prompt
        self.logger.info("System prompt updated")

    def clear_system_prompt(self) -> None:
        """
        Remove the system prompt.
        """
        self.system_prompt = None
        self.logger.info("System prompt cleared")
