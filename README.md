# KEIR: Knowledge-Enhanced Iterative Reasoning

Reference implementation accompanying **Resource-Efficient Mathematical Problem
Solving with Lower-Capacity Language Models via Knowledge-Enhanced Iterative
Reasoning**, by Xingnan Jin, Yongping Du, and Honggui Han.

KEIR constructs answer-equivalent question variants and aligned subtask–solution
chains, trains planning and execution adapters on a frozen backbone, and solves
problems by semantic consolidation followed by sequential local execution.

This source release includes method code, prompt templates, preprocessing,
training configurations for seven backbone families, inference, answer scoring,
statistical evaluation, and unit tests. See
[release scope and requirements](docs/reproducibility.md).
