"""
Entropy Lab — a small, honest Streamlit app for building intuition about
token-level uncertainty in LLMs: softmax, Shannon entropy, surprisal,
and a simple (illustrative, not validated) risk score built from them.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="Entropy Lab", layout="wide")

# ---------------------------------------------------------------------------
# Model loading (cached so switching UI controls doesn't reload every time)
# ---------------------------------------------------------------------------

MODEL_OPTIONS = ["distilgpt2", "gpt2", "gpt2-medium"]


@st.cache_resource(show_spinner="Loading model...")
def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return tok, model


# ---------------------------------------------------------------------------
# Core math: for one generation step, given logits over the vocabulary,
# compute the full softmax distribution, its Shannon entropy, and the
# surprisal (negative log-probability) of whichever token gets chosen.
# ---------------------------------------------------------------------------

def step_stats(logits: torch.Tensor, chosen_id: int, top_k: int = 8):
    """logits: 1D tensor of size [vocab]. Returns a dict of everything the
    UI needs to both show a number AND show the arithmetic behind it."""
    probs = F.softmax(logits, dim=-1)                      # p_i = softmax(z)_i
    log_probs = F.log_softmax(logits, dim=-1)               # ln p_i, computed stably

    # Shannon entropy over the FULL vocabulary: H = -sum_i p_i * ln(p_i)
    entropy_nats = -(probs * log_probs).sum().item()
    entropy_bits = entropy_nats / math.log(2)

    # Surprisal of the token that was actually chosen: -ln p(chosen)
    surprisal_nats = -log_probs[chosen_id].item()

    # Top-k breakdown, for the "show your work" panel
    top_probs, top_ids = torch.topk(probs, top_k)
    top_rows = []
    explained_mass = 0.0
    explained_entropy = 0.0
    for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
        contrib = -p * math.log(p) if p > 0 else 0.0
        explained_mass += p
        explained_entropy += contrib
        top_rows.append({"token_id": tid, "p": p, "contribution": contrib})

    return {
        "probs": probs,
        "entropy_nats": entropy_nats,
        "entropy_bits": entropy_bits,
        "surprisal_nats": surprisal_nats,
        "top_rows": top_rows,
        "explained_mass": explained_mass,
        "explained_entropy": explained_entropy,
        "chosen_prob": probs[chosen_id].item(),
    }


# ---------------------------------------------------------------------------
# Manual, step-by-step generation loop (deliberately not model.generate(),
# so every step's logits are visible and nothing is hidden inside a library
# call).
# ---------------------------------------------------------------------------

def generate_with_stats(tok, model, prompt: str, max_new_tokens: int,
                          strategy: str, temperature: float, top_k_sample: int):
    input_ids = tok.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()
    records = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            out = model(generated)
            logits = out.logits[0, -1, :]  # logits for the next token, shape [vocab]

            if strategy == "Greedy":
                chosen_id = int(torch.argmax(logits).item())
            else:
                scaled = logits / max(temperature, 1e-5)
                probs = F.softmax(scaled, dim=-1)
                topk_probs, topk_ids = torch.topk(probs, top_k_sample)
                topk_probs = topk_probs / topk_probs.sum()
                pick = torch.multinomial(topk_probs, 1).item()
                chosen_id = int(topk_ids[pick].item())

            stats = step_stats(logits, chosen_id)
            token_str = tok.decode([chosen_id])
            records.append({
                "step": step,
                "token": token_str,
                "token_id": chosen_id,
                "prob": stats["chosen_prob"],
                "surprisal_nats": stats["surprisal_nats"],
                "entropy_nats": stats["entropy_nats"],
                "entropy_bits": stats["entropy_bits"],
                "top_rows": stats["top_rows"],
                "explained_mass": stats["explained_mass"],
                "explained_entropy": stats["explained_entropy"],
            })

            generated = torch.cat([generated, torch.tensor([[chosen_id]])], dim=1)
            if tok.eos_token_id is not None and chosen_id == tok.eos_token_id:
                break

    return records


# ---------------------------------------------------------------------------
# A simple, clearly-labeled heuristic risk score. This is NOT a validated
# hallucination detector — it exists to build intuition about how single-pass
# signals combine. The actual project should benchmark real Tier-1/Tier-2
# signals against labeled data before trusting any such score.
# ---------------------------------------------------------------------------

def risk_score(records, high_entropy_threshold: float):
    if not records:
        return None
    mean_entropy = np.mean([r["entropy_nats"] for r in records])
    mean_surprisal = np.mean([r["surprisal_nats"] for r in records])
    frac_high = np.mean([r["entropy_nats"] > high_entropy_threshold for r in records])

    score = (
        0.5 * min(mean_entropy / 5.0, 1.0)
        + 0.3 * min(mean_surprisal / 8.0, 1.0)
        + 0.2 * frac_high
    ) * 100
    return {
        "score": score,
        "mean_entropy": mean_entropy,
        "mean_surprisal": mean_surprisal,
        "frac_high": frac_high,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Entropy Lab")
st.caption(
    "Watch a real language model generate text one token at a time, and see "
    "exactly how the softmax/entropy/surprisal numbers behind each token are computed."
)

with st.sidebar:
    st.header("Setup")
    model_name = st.selectbox("Model", MODEL_OPTIONS, index=0,
                               help="Small models are used by default so this runs on CPU.")
    prompt = st.text_area("Prompt", "The capital of France is")
    max_new_tokens = st.slider("Tokens to generate", 1, 60, 20)
    strategy = st.radio("Decoding", ["Greedy", "Sampling"], horizontal=True)
    temperature, top_k_sample = 1.0, 10
    if strategy == "Sampling":
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        top_k_sample = st.slider("Top-k", 1, 50, 10)
    high_entropy_threshold = st.slider(
        "\"High entropy\" cutoff (nats)", 0.5, 5.0, 2.0, 0.1,
        help="Used only for the illustrative risk score and the heatmap coloring."
    )
    run = st.button("Generate", type="primary")

if run:
    tok, model = load_model(model_name)
    with st.spinner("Generating..."):
        records = generate_with_stats(
            tok, model, prompt, max_new_tokens, strategy, temperature, top_k_sample
        )
    st.session_state["records"] = records
    st.session_state["threshold"] = high_entropy_threshold

if "records" not in st.session_state:
    st.info("Set a prompt in the sidebar and click Generate.")
    st.stop()

records = st.session_state["records"]
threshold = st.session_state.get("threshold", high_entropy_threshold)

# --- Colored output text, heatmapped by entropy ---
st.subheader("Generated text, colored by uncertainty")
st.caption("Redder = higher entropy at that step (the model was less sure what should come next).")

def entropy_color(e, vmax=4.0):
    t = min(e / vmax, 1.0)
    r = int(230 * t + 20)
    g = int(230 * (1 - t) + 20)
    return f"rgb({r},{g},60)"

spans = []
for r in records:
    color = entropy_color(r["entropy_nats"])
    safe_tok = r["token"].replace("<", "&lt;").replace(">", "&gt;")
    spans.append(
        f'<span title="entropy={r["entropy_nats"]:.2f} nats, p={r["prob"]:.3f}" '
        f'style="background:{color};padding:2px 1px;border-radius:2px;">{safe_tok}</span>'
    )
st.markdown(f"<div style='font-size:20px; line-height:2.1'>{prompt}{''.join(spans)}</div>",
            unsafe_allow_html=True)

# --- Risk score ---
st.subheader("Illustrative risk score")
st.warning(
    "This score is a simple, hand-weighted combination of raw signals for building "
    "intuition only. It has not been validated against labeled hallucination data — "
    "do not treat it as a real detector. The actual project benchmarks Tier-1/Tier-2 "
    "signals properly against ground truth."
)
rs = risk_score(records, threshold)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk score (0-100)", f"{rs['score']:.1f}")
c2.metric("Mean entropy", f"{rs['mean_entropy']:.2f} nats")
c3.metric("Mean surprisal", f"{rs['mean_surprisal']:.2f} nats")
c4.metric("Fraction high-entropy tokens", f"{rs['frac_high']*100:.0f}%")

# --- Per-token chart ---ƒ
st.subheader("Entropy and surprisal across the generation")
df = pd.DataFrame([{
    "step": r["step"], "token": r["token"],
    "entropy_nats": r["entropy_nats"], "surprisal_nats": r["surprisal_nats"],
} for r in records]).set_index("step")
st.line_chart(df[["entropy_nats", "surprisal_nats"]])
st.dataframe(
    df.rename(columns={"entropy_nats": "entropy (nats)", "surprisal_nats": "surprisal (nats)"}),
    width="stretch",
)

# --- The math, worked out for a chosen step ---
st.subheader("Show the math for one token")
step_idx = st.slider("Pick a generation step to inspect", 0, len(records) - 1, 0)
r = records[step_idx]
tok_obj = load_model(model_name)[0]

st.markdown(f"**Token generated at step {step_idx}:** `{r['token']}`  (chosen with probability {r['prob']:.4f})")

colA, colB = st.columns(2)
with colA:
    st.markdown("**1. Softmax turns logits into probabilities**")
    st.latex(r"p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}")
    st.markdown("**2. Shannon entropy over the full vocabulary**")
    st.latex(r"H = -\sum_{i=1}^{V} p_i \ln p_i")
    st.markdown(
        f"Computed over all {tok_obj.vocab_size:,} vocabulary entries: "
        f"**H = {r['entropy_nats']:.3f} nats** ({r['entropy_bits']:.3f} bits)."
    )
    st.markdown("**3. Surprisal of the token actually chosen**")
    st.latex(r"I(w) = -\ln p(w)")
    st.markdown(f"**I = {r['surprisal_nats']:.3f} nats** for the chosen token.")

with colB:
    st.markdown("**Worked example — top contributions to H**")
    st.caption("Entropy sums over the whole vocabulary; here are the top 8 terms, which usually explain most of it when the model is confident.")
    rows = []
    for row in r["top_rows"]:
        tok_str = tok_obj.decode([row["token_id"]])
        rows.append({
            "token": tok_str,
            "p_i": round(row["p"], 4),
            "-p_i * ln(p_i)": round(row["contribution"], 4),
        })
    st.table(pd.DataFrame(rows))
    st.markdown(
        f"Sum of these 8 terms: **{r['explained_entropy']:.3f} nats**, covering "
        f"**{r['explained_mass']*100:.1f}%** of the probability mass. "
        f"The remaining **{r['entropy_nats'] - r['explained_entropy']:.3f} nats** "
        f"comes from the long tail of the other {tok_obj.vocab_size - 8:,} tokens, "
        f"each with tiny but nonzero probability."
    )

st.caption(
    "Entropy tells you how spread out the model's beliefs were *before* it picked a token. "
    "Surprisal tells you how unlikely the token it actually picked was. A model can be "
    "low-entropy (confident) and still pick a wrong, high-surprisal answer if it's confidently wrong — "
    "that's the dangerous case worth watching for."
)
