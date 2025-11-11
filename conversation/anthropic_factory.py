import os
from typing import Optional

from conversation.anthropic_conversation import AnthropicConversation


class AnthropicFactory:
    """
    Factory for creating AnthropicConversation instances.
    
    Manages API client initialization and provides a clean interface for
    creating conversation objects with consistent configuration.
    """

    def __init__(
            self,
            model_name: str,
            system_prompt: Optional[str] = None,
            max_tokens: int = 1000,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None
    ):
        """
        Initialize the Anthropic conversation factory.
        
        Args:
            model_name: Name of the Claude model (e.g., "anthropic-claude-3-opus-20240229")
            system_prompt: Optional system prompt for all conversations from this factory
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 to 1.0). Optional, uses API default if not set.
                        Note: Anthropic API only allows one of temperature or top_p.
            top_p: Nucleus sampling parameter (0.0 to 1.0). Optional, uses API default if not set.
                   Note: Anthropic API only allows one of temperature or top_p.
        """
        # Anthropic API constraint: cannot specify both temperature and top_p
        if temperature is not None and top_p is not None:
            raise ValueError(
                "Anthropic API does not allow both 'temperature' and 'top_p' to be specified. "
                f"Got temperature={temperature} and top_p={top_p}. "
                "Please use only one or leave both as None to use API defaults."
            )

        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_AI_KEY"])

        # Remove "anthropic-" prefix if present
        model_name = model_name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def get_conversation(self) -> AnthropicConversation:
        """
        Create a new conversation instance.
        
        Returns:
            AnthropicConversation configured with factory settings
        """
        return AnthropicConversation(
            client=self.client,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
