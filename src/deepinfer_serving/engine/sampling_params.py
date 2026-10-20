class SamplingParams:
    """Subset of upstream ``deepinfer/sampling_params.py``. Only greedy decoding is implemented."""

    def __init__(
        self,
        temperature: float = 1.0,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        eos_id: int = 0,
    ) -> None:
        if temperature != 0.0:
            raise ValueError("mini-DeepInfer implements greedy decoding only; pass temperature=0.0")
        self.temperature = temperature
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens if max_tokens != 0 else 1
        self.eos_id = eos_id

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "ignore_eos": self.ignore_eos,
            "max_tokens": self.max_tokens,
            "eos_id": self.eos_id,
        }
