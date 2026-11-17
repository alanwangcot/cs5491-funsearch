# Reasoning Augmented FunSearch for Online Bin Packing

We compare code-only program search with a version that also evolves a short strategy description, using DeepSeek V4.1 Flash. Both start from First Fit and Best Fit and share one packing evaluator.

## Setup

Requires Python 3.11+, uv and Docker.

```bash
uv sync --frozen
docker build -f Dockerfile.evaluator -t funsearch-evaluator:1 .
uv run pytest -q
```

## Evaluate the milestone work

The prepared datasets and the 16-call pilot are included. These commands need no API key:

```bash
uv run python -m funsearch_project.cli data prepare
uv run python -m funsearch_project.cli baselines
uv run python -m funsearch_project.cli smoke --provider fake
uv run python scripts/evaluate_pilot.py
```

Both pilot methods produced eight valid programs and a best bin-use ratio of 1.000 relative to Best Fit. The PDF and LaTeX source are in `milestone/`.

To run a new short search, set `DEEPSEEK_API_KEY` and run:

```bash
uv run python -m funsearch_project.cli pilot --config configs/development.yaml
```

## Layout

- `src/funsearch_project/`: evaluator, API interface and search code.
- `configs/` and `prompts/`: settings and model prompts.
- `data/`: benchmark instances and recorded pilot programs.
- `tests/`: packing, generation and search tests.
- `milestone/`: milestone PDF and LaTeX source.

The synthetic data use Weibull shape 3, scale 45, 500 items and capacity 100. The additional benchmark is [OR-Library binpack3](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html). The search approach follows [FunSearch](https://doi.org/10.1038/s41586-023-06924-6).
