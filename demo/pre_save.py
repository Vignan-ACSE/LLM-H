import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_OPTIONS = [
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

PROMPTS = {
    "High Confidence (Solid Fact)": (
        "The capital of France is Paris.",
        MODEL_OPTIONS[0],
    ),
    "The Wild Guess (Complete Hallucination)": (
        "The exact atomic weight of undiscovered element 130 is exactly 320.5.",
        MODEL_OPTIONS[0],
    ),
    "The Ambiguous Case (Synonym Split)": (
        "The President preferred to drink Folgers coffee every morning.",
        MODEL_OPTIONS[0],
    ),
    "Vignan University: AIML Dean": (
        "The Dean of the School of Computing and Informatics at Vignan University is",
        MODEL_OPTIONS[0],
    ),
    "Vignan University: AIML Curriculum": (
        "Describe the Artificial Intelligence and Machine Learning (AIML) program curriculum at Vignan University.",
        MODEL_OPTIONS[1],
    ),
}


def analyze(prompt, model_name, max_new_tokens=50):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = prompt

    input_ids = tok.encode(formatted, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )

    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
    plain_text = tok.decode(generated_ids, skip_special_tokens=True)

    records = []

    for step, token_id in enumerate(generated_ids):
        logits = outputs.scores[step][0]
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        chosen_id = int(token_id.item())
        chosen_prob = probs[chosen_id].item()
        chosen_logit = logits[chosen_id].item()
        entropy = -(probs * log_probs).sum().item()
        surprisal = -log_probs[chosen_id].item()

        top_probs, top_ids = torch.topk(probs, min(8, probs.shape[-1]))

        top_rows = []
        explained_mass = 0.0
        explained_entropy = 0.0

        for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
            contribution = -p * math.log(p) if p > 0 else 0.0
            explained_mass += p
            explained_entropy += contribution
            top_rows.append({
                "token_id": int(tid),
                "p": p,
                "contribution": contribution,
                "token_str": tok.decode([int(tid)]),
            })

        records.append({
            "step": step,
            "token": tok.decode([chosen_id]),
            "token_id": chosen_id,
            "prob": chosen_prob,
            "surprisal_nats": surprisal,
            "entropy_nats": entropy,
            "entropy_bits": entropy / math.log(2),
            "top_rows": top_rows,
            "explained_mass": explained_mass,
            "explained_entropy": explained_entropy,
            "vocab_size": int(logits.shape[-1]),
            "chosen_logit": chosen_logit,
        })

    del model
    del tok

    return {
        "prompt": prompt,
        "model_name": model_name,
        "plain_text": plain_text,
        "records": records,
    }


def main():
    results = {}

    for key, (prompt, model_name) in PROMPTS.items():
        print(f"Generating: {key}")
        results[key] = analyze(prompt, model_name)

    Path("pre_saved_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Created pre_saved_results.json")


if __name__ == "__main__":
    main()
