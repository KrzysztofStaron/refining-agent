"""Run the brain-optimizer pipeline: generate post variants, score each with
TRIBE v2, and save everything to results.json. The PNG is then drawn from that
JSON by render.py (which needs no GPU), so you can re-style the figure without
re-running the model.

Usage:
    python visualize.py
"""
import tempfile
import os
import json
import requests
import numpy as np
import neuralset.extractors.text as _t

OPENROUTER_TOKEN = "YOUR_TOKEN_HERE"
N_VARIANTS = 4
ORIGINAL_POST = "Just shipped a new feature that will change how you think about productivity. Thread 🧵"
VIEWS = ["left", "right", "dorsal"]
RESULTS_JSON = "results.json"
OUT_PNG = "brain_optimizer.png"


def _patched_load(self):
    from transformers import AutoModel, AutoModelForCausalLM
    is_causal = any(x in self.model_name for x in ["Llama", "Phi-3", "gpt", "opt"])
    Model = AutoModelForCausalLM if is_causal else AutoModel
    model = Model.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")
    model.eval()
    return model

_t.HuggingFaceText._load_model = _patched_load


def generate_variants(post: str, n: int) -> list[str]:
    print(f"Generating {n} variants via OpenRouter...")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_TOKEN}"},
        json={
            "model": "anthropic/claude-haiku-4-5",
            "messages": [{"role": "user", "content": (
                f"Generate {n} different variants of this X (Twitter) post. "
                "Each variant should have a different tone, hook, or structure "
                "but convey the same core message. "
                "Return ONLY the variants, one per line, no numbering, no extra text.\n\n"
                f"Original post:\n{post}"
            )}],
        },
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    return [line.strip() for line in text.strip().splitlines() if line.strip()][:n]


def get_activation(post: str, tribe) -> tuple[np.ndarray, float]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(post)
        tmp_path = f.name
    try:
        df = tribe.get_events_dataframe(text_path=tmp_path)
        preds, _ = tribe.predict(events=df, verbose=False)
        avg = preds.mean(axis=0)
        score = float(np.mean(np.abs(avg)))
        return avg, score
    finally:
        os.unlink(tmp_path)


def main():
    from tribev2 import TribeModel

    print("Loading TRIBE v2...")
    tribe = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

    variants = generate_variants(ORIGINAL_POST, N_VARIANTS)
    all_posts = [ORIGINAL_POST] + variants
    labels = ["ORIGINAL"] + [f"VARIANT {i+1}" for i in range(len(variants))]

    print("Scoring posts...")
    results = []
    for label, post in zip(labels, all_posts):
        print(f"  {label}...")
        avg, score = get_activation(post, tribe)
        results.append({"label": label, "post": post, "score": score, "activation": avg})

    # rank by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # serialize: activations become plain lists so the JSON is self-contained
    payload = {
        "original_post": ORIGINAL_POST,
        "views": VIEWS,
        "results": [
            {
                "rank": r["rank"],
                "label": r["label"],
                "post": r["post"],
                "score": r["score"],
                "activation": r["activation"].tolist(),
            }
            for r in results
        ],
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f)
    print(f"\nSaved {RESULTS_JSON} ({len(results)} posts)")

    # draw the PNG from the JSON we just wrote
    import render
    render.render(RESULTS_JSON, OUT_PNG)


if __name__ == "__main__":
    main()
