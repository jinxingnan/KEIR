# Third-party material

The MATH answer-normalization routines in `src/keir/answers.py` are adapted
from [hendrycks/math](https://github.com/hendrycks/math), specifically
`modeling/math_equivalence.py`. Copyright (c) 2021 Dan Hendrycks.
The original MIT license is retained in `licenses/hendrycks-math-MIT.txt`.
Those routines remain available under MIT; the KEIR research restriction
does not replace their original permission.

PyTorch, Transformers, PEFT, Accelerate, Datasets, NumPy, PyYAML, the OpenAI
SDK, and bitsandbytes are external dependencies, not vendored source. Their
licenses govern their respective components. Install them from their maintainers.

SimCSE-RoBERTa is loaded from `princeton-nlp/sup-simcse-roberta-base`.
See [SimCSE](https://github.com/princeton-nlp/SimCSE) for the encoder and citation.
Backbone checkpoints are downloaded separately and require compliance with
their own model licenses and access conditions. The KEIR license does not
grant rights to Llama, Qwen, DeepSeek, Gemma, RoBERTa or their checkpoints.

No benchmark dataset, teacher-generated training corpus or model weights are
redistributed in this source package. Dataset source links and actual split
names are in `metadata/benchmarks.json`. In particular, the ASDiv release
identifies CC BY-NC 4.0 terms; the KEIR code license does not override them.
