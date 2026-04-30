import os
from vllm import LLM, SamplingParams


def generate(
    prompt: str,
    model: str = "embedl/Qwen3-1.7B-FlashHead-W4A16",
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate text using a FlashHead-optimized model via vLLM."""
    llm = LLM(model=model)
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text


if __name__ == "__main__":
    result = generate("Explain quantum entanglement in one paragraph.")
    print(result)