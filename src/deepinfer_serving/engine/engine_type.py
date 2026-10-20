import enum


class EngineType(enum.Enum):
    """LLMEngine type. Mirrors upstream ``deepinfer/engine/engine_type.py``."""

    NORMAL = 0  # process prompts & completions
    PROMPT = enum.auto()  # only process prompts
    COMPLETION = enum.auto()  # only process completions
