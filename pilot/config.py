"""Pilot configuration: every key the system reads, its default, and why it exists.

The brief guarantees that ``config`` contains at least ``model`` and ``api_key_env``;
every other key falls back to ``DEFAULTS``. ``KEY_DOCS`` explains each key in a sentence
or two (the README's configuration table comes from it), and a test checks that every
default is documented.
"""

DEFAULTS: dict = {
    "model": "qwen/qwen3.7-flash",
    "api_key_env": "OPENROUTER_API_KEY",
    "base_url": "https://openrouter.ai/api/v1",
    "helper_model": None,
    "memory": "full",
    "llm": "openrouter",
    "max_input_tokens": 8000,
    "target_input_tokens": 7400,
    "reply_max_tokens": 900,
    "helper_max_tokens": 1500,
    "temperature": 0.3,
    "seed": 7,
    "reasoning": False,
    "request_timeout_s": 60,
    "max_retries": 3,
    "log_prompts": False,
    "provider": None,
}

KEY_DOCS: dict[str, str] = {
    "model": (
        "OpenRouter id of the model that writes the user-facing reply. It lives in config, "
        "not code, because the graders run our code with their own key and model choice."
    ),
    "api_key_env": (
        "Name of the environment variable that holds the API key. It is read at every call, "
        "so it may be set after the Assistant is built, and stripped of stray whitespace. "
        "The key itself never appears in config, code, logs or error messages."
    ),
    "base_url": (
        "OpenAI-compatible API root; requests go to <base_url>/chat/completions. The "
        "OpenRouter-only fields (usage, reasoning, provider) are sent only when it is openrouter.ai."
    ),
    "helper_model": (
        "Model for background helper calls (turning what the user said into memory records). "
        "None means: use `model`. Helper calls do not count toward the reply token cap."
    ),
    "memory": (
        "'full' is the complete design (keyword search plus extracted account records); "
        "'lex' is the same system without the extractor, to measure what extraction adds."
    ),
    "llm": (
        "'openrouter' makes real API calls; 'fake' swaps in FakeLLM (no network, no cost) "
        "for tests and offline runs."
    ),
    "max_input_tokens": (
        "The brief's hard cap on input tokens for every call that produces a reply. "
        "A reply prompt is never sent above it."
    ),
    "target_input_tokens": (
        "What the prompt builder aims for, using our local token count. The gap below "
        "max_input_tokens absorbs any difference between our count and the provider's."
    ),
    "reply_max_tokens": (
        "Output cap for a reply: long enough for a drafted email or meeting prep, "
        "short enough to bound cost and latency."
    ),
    "helper_max_tokens": (
        "Output cap for one helper (extraction) call. At 700, 5 of 2,896 answers in the final "
        "runs were cut off mid-JSON on long messages and their notes were lost."
    ),
    "temperature": (
        "Sampling temperature for replies. Low, because users want consistent, factual "
        "answers about their accounts rather than creative variety."
    ),
    "seed": (
        "Sampling seed sent with every call, so evaluation reruns are as repeatable as "
        "the provider allows (not every provider honours it)."
    ),
    "reasoning": (
        "False sends reasoning={\"enabled\": false} so the model skips hidden 'thinking' "
        "tokens (faster and cheaper). A model that refuses to switch reasoning off is "
        "retried once without the field, automatically."
    ),
    "request_timeout_s": (
        "Seconds to wait on an HTTP request before it counts as a timeout "
        "(timeouts are retried like other temporary failures)."
    ),
    "max_retries": (
        "How many times a rate-limited (429), server-side (5xx), timed-out or dropped "
        "request is retried, waiting 2, 4, then 8 seconds between tries."
    ),
    "log_prompts": (
        "When True, every reply prompt is appended to <data_dir>/prompts.jsonl. The eval "
        "turns it on to check whether the needed fact reached the prompt; off by default "
        "because it writes a second copy of user data to disk."
    ),
    "provider": (
        "Optional OpenRouter provider-routing object, for example "
        "{\"order\": [\"<host>\"], \"allow_fallbacks\": false}, passed through unchanged "
        "to pin which upstream host serves the model."
    ),
}


def load_config(config: dict | None) -> dict:
    """Return ``DEFAULTS`` overlaid with ``config``.

    Always returns a new dict, so neither ``DEFAULTS`` nor the caller's dict can be
    changed by accident. Unknown keys are kept and ignored, so a caller can pass a larger
    settings dict (eval/factory.py: make_system adds ``memory`` to the eval's system config).
    """
    merged = dict(DEFAULTS)
    if config:
        merged.update(config)
    return merged
