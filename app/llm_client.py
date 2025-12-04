import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, RateLimitError, NotFoundError

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Please add it to your environment or .env file.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

_models_env = os.getenv("OPENROUTER_MODEL_LIST", "").strip()
if _models_env:
    FREE_MODELS: List[str] = [m.strip() for m in _models_env.split(",") if m.strip()]
else:
    FREE_MODELS = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b:free",
        "google/gemma-3-12b:free",
        "mistralai/mistral-7b-instruct:free",
        "deepseek/deepseek-r1-distill-llama-70b:free",
        "nousresearch/hermes-3-405b:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "alibaba/tongyi-deepresearch-30b-a3b:free",
    ]

if not FREE_MODELS:
    raise RuntimeError("No models configured. Please set OPENROUTER_MODEL_LIST in your .env file.")


def _try_model(model: str, system_prompt: str, text: str) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=800,
            temperature=0.2,
        )
        out = (response.choices[0].message.content or "").strip()
        return out or None
    except (RateLimitError, APIError, NotFoundError) as e:
        print(f"[LLM ERROR] model={model} error={e}")
        return None


def improve_text(
    text: str,
    tone: str = "neutral professional",
    language: str = "en",
) -> str:
    """
    Improve writing using a chain of free OpenRouter models.
    """

    system_prompt = f"""
You are a STRICT grammar and style editor for short texts.

Your job:
- Take messy chat-style English or Hinglish.
- Return a clean, well-written version with correct spelling, grammar, punctuation and capitalization.

Hard rules:
1) ALWAYS fix:
   - Spelling
   - Grammar and word order
   - Capitalization
   - Punctuation (commas, full stops, question marks, etc.).
2) Split run-on sentences into clear separate sentences if needed.
3) NEVER start the answer with explanations like "I think", "It seems", or "You meant".
   Output MUST be only the corrected text.
4) Preserve the original meaning and emotional tone (casual, friendly, angry, funny, etc.).
5) Keep Hinglish flavour words (kya, kyun, kaise, yaar, bhai, nahi, etc.) when they add style,
   but make the overall sentence grammatically correct.
6) Expand common chat abbreviations:
   - cn  -> can
   - cnt -> can't
   - idk -> I don't know
   - btw -> by the way
   - ppl -> people
   - u   -> you
   - ur  -> your
   (and similar obvious cases).
7) Remove obvious keyboard-smash gibberish (like "Khdbwckhwdbcmwdncii") that has no meaning.
8) Do NOT add new ideas or information. Just polish what is there.
9) Output language should primarily be {language}, but Hinglish words are allowed.
10) Return ONLY the corrected text. No quotes, no extra commentary.

Examples (follow the style very closely):

Input:  fx ths it nt gud txt , it nomal
Output: Fix this; it is not good text, it is just normal.

Input:  I cn do t bt kya jo main nhi kr paa rha idk yaar
Output: I can do it, but I don't know why I am not able to do some things, yaar.

Input:  plz tel me kya ho rha h idk mn
Output: Please tell me what is happening; I don't know, man.

Input:  Khdbwckhwdbcmwdncii I cn do t bt what i cnt idk mn
Output: I can do it, but I don't know what I can't do, man.

Tone preference: {tone}.
"""

    print(f"[LLM] improve_text called. FREE_MODELS={FREE_MODELS}")
    for model in FREE_MODELS:
        print(f"[LLM] Trying model: {model}")
        out = _try_model(model, system_prompt, text)
        if out:
            print(f"[LLM] Model {model} succeeded. Done.")
            return out

    print("[LLM] All configured models failed or returned empty. Falling back to original text.")
    return text

