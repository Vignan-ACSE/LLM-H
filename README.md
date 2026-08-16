# LLM-H

## 1. Topics to Learn
### 1.1 Transformer
1. Attention is all you need [Paper](https://arxiv.org/pdf/1706.03762v1)
2. The Illustrated Transformer [Blog](https://jalammar.github.io/illustrated-transformer/)
3. **Coding Session:** Try write code for Transformer as simple as possible by using OOPS and Python

**Follow-up works**
`MUST READ`

- [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198)
- [One Model To Learn Them All](https://arxiv.org/pdf/1706.05137)
- [Image Transformer](https://arxiv.org/pdf/1802.05751)
- [Training Tips for the Transformer Model](https://arxiv.org/pdf/1804.00247)

`IF Interested`
- [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/pdf/1706.03059)
- [Discrete Autoencoders for Sequence Models](https://arxiv.org/pdf/1801.09797)
- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
- [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/pdf/1803.03382)
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/pdf/1804.04235)
### 1.2 Pytorch
1. Follow or learn from official documentation [website](https://docs.pytorch.org/tutorials/)
2. Learn Pytorch blogs [website](https://www.learnpytorch.io/)
3. Learn Pytorch with notebooks [github](https://github.com/mrdbourke/pytorch-deep-learning)
### 1.3 Setting up a Research Repo
For a **research GitHub repo**, keep it simple but reproducible:

#### Must-have

* `README.md` — project overview, setup, usage, results
* `LICENSE` — e.g. MIT/Apache-2.0
* `requirements.txt` or `environment.yml` — dependencies
* `conda` or `pip` — Virtual environment
* `.gitignore` — Python/IDE/cache/model files
* `src/` — main source code
* `configs/` — experiment logs/config files
* `scripts/` — training, evaluation, preprocessing scripts
* `tests/` — basic tests
* `data/` — **only small/sample data**, not huge datasets
* `checkpoints/` — usually ignored by Git; document where to download models
* `results/` — experiment outputs, tables, figures
* `docs/` — detailed methodology/notes if needed

#### Research-specific

* `paper/` — LaTeX paper
* `experiments/` — experiment logs/configs
* `baselines/` — baseline implementations
* `notebooks/` — exploratory notebooks
* `CHANGELOG.md` — important changes
* `CITATION.cff` — how others should cite your work

#### Reproducibility

* Fixed random seeds
* Exact dependency versions
* Dataset download/preprocessing instructions
* Model/config files
* Training command
* Evaluation command
* Hardware requirements
* Experiment results
* Git tags/releases for paper versions


For your research projects, I'd especially prioritize **`configs/`, `scripts/`, experiment tracking, reproducibility, and `CITATION.cff`**.

## 2. LLMs
### 2.1 Things to Learn
1. [[1hr Talk] Intro to Large Language Models](https://youtu.be/zjkBMFhNj_g?si=A06tQlmpfvOwuW8s) - By the founding member of OpenAI **andrej karpathy**
2. [PPT of the above talk](https://drive.google.com/file/d/1pxx_ZI7O-Nwl7ZLNk5hI3WzAsTLwvNU7/view)

### 2.2 Types of Hallucinations
#### 1. Intrinsic Hallucination

**The generated answer contradicts the information provided in the source, context, or input.**

* **Contextual hallucination** — The answer contradicts or changes information given in the prompt/context.
* **Logical hallucination** — The reasoning or conclusion contradicts facts or relationships established in the given context.
* **Semantic hallucination** — The model changes the meaning of the provided information.
* **Instruction hallucination** — The model misunderstands or violates information/instructions contained in the input.

**Example:**
Context: *“John is 25 years old.”*
LLM: *“John is 32 years old.”*
→ **Intrinsic hallucination**

#### 2. Extrinsic Hallucination

**The generated answer introduces information that cannot be supported by the provided source or context.**

* **Factual hallucination** — Generates facts that are incorrect or unsupported.
* **Fabricated information** — Invents people, papers, events, statistics, etc.
* **Citation hallucination** — Creates fake or incorrect references.
* **Source hallucination** — Claims that a source supports something when it does not.
* **Temporal hallucination** — Provides outdated or unsupported information about current events.
* **Entity hallucination** — Invents or incorrectly describes entities.
* **Commonsense hallucination** — Produces claims that conflict with general real-world knowledge.
* **Mathematical hallucination** — Produces unsupported or incorrect numerical results.
* **Code hallucination** — Invents APIs, libraries, functions, or technical details.
* **Multimodal hallucination** — Describes objects, attributes, or relationships that are not present in an image/video.

**Example:**
Context: *“John is a software engineer.”*
LLM: *“John works at Google and has 10 years of experience.”*
If the context provides no evidence for these claims → **Extrinsic hallucination**

#### Simple distinction

| Type          | Meaning                                  |
| ------------- | ---------------------------------------- |
| **Intrinsic** | **Contradicts** the provided information |
| **Extrinsic** | **Adds unsupported** information         |

A useful research formulation is:

> **Intrinsic hallucination = contradiction with the source/context.**
> **Extrinsic hallucination = information not entailed or supported by the source/context.**

One important point: **factual, mathematical, code, citation, and other categories are dimensions/types of hallucination, while intrinsic vs. extrinsic describes their relationship to the available evidence.** So the categories can overlap rather than being a perfectly exclusive taxonomy.


## 3. Project

## 4. Publication
