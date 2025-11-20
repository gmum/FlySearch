from enum import Enum

from conversation.anthropic_factory import AnthropicFactory
from conversation.gpt_factory import GPTFactory
from conversation.vllm_factory import VLLMFactory
from conversation.gemini_factory import GeminiFactory


class LLMBackends(str, Enum):
    VLLM = "vllm"
    GPT = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


LLM_BACKEND_FACTORIES = {
    LLMBackends.VLLM: VLLMFactory,
    LLMBackends.GPT: GPTFactory,
    LLMBackends.ANTHROPIC: AnthropicFactory,
    LLMBackends.GEMINI: GeminiFactory,
}
