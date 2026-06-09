import tempfile
import os
import requests
import numpy as np
import neuralset.extractors.text as _t

OPENROUTER_TOKEN = "YOUR_TOKEN_HERE"
N_VARIANTS = 5
ORIGINAL_POST = "Just shipped a new feature that will change how you think about productivity. Thread 🧵"


def _patched_load(self):
    from transformers import AutoModel, AutoModelForCausalLM
    is_causal = any(x in self.model_name for x in ["Llama", "Phi-3", "gpt", "opt"])
    Model = AutoModelForCausalLM if is_causal else AutoModel
    model = Model.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")
    model.eval()
    return model

_t.HuggingFaceText._load_model = _patched_load


def generate_variants(post: str, n: int) -> list[str]:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_TOKEN}"},
        json={
            "model": "anthropic/claude-haiku-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Generate {n} different variants of this X (Twitter) post. "
                        "Each variant should have a different tone, hook, or structure "
                        "but convey the same core message. "
                        "Return ONLY the variants, one per line, no numbering, no extra text.\n\n"
                        f"Original post:\n{post}"
                    ),
                }
            ],
        },
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    variants = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return variants[:n]


def score_post(post: str, model) -> float:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(post)
        tmp_path = f.name
    try:
        df = model.get_events_dataframe(text_path=tmp_path)
        preds, _ = model.predict(events=df, verbose=False)
        return float(np.mean(np.abs(preds)))
    finally:
        os.unlink(tmp_path)


def main():
    from tribev2 import TribeModel

    print("Loading TRIBE v2...")
    tribe = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

    print(f"\nOriginal post:\n  {ORIGINAL_POST}\n")
    print(f"Generating {N_VARIANTS} variants via OpenRouter...")
    variants = generate_variants(ORIGINAL_POST, N_VARIANTS)

    all_posts = [ORIGINAL_POST] + variants
    print(f"\nScoring {len(all_posts)} posts with TRIBE v2...\n")

    scores = []
    for i, post in enumerate(all_posts):
        label = "ORIGINAL" if i == 0 else f"VARIANT {i}"
        score = score_post(post, tribe)
        scores.append((score, label, post))
        print(f"[{label}] score={score:.4f}\n  {post}\n")

    scores.sort(reverse=True)
    best_score, best_label, best_post = scores[0]
    print("=" * 60)
    print(f"WINNER: {best_label} (score={best_score:.4f})")
    print(f"  {best_post}")


if __name__ == "__main__":
    main()
