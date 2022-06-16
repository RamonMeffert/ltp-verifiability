# A Comparison of Prompt Engineering Methods for Checking Verifiability of Online User Comments

Md Zobaer Hossain, Linfei Zhang, Robert van Timmeren and Ramon Meffert, June 2022

This repository contains the source code for the experiments, data processing
and data analysis conducted as part of our course project for the 2021–2022
edition of the _Language Technology Project_ course at the University of
Groningen.

## Data

All files related to the datasets are located in the [datasets](datasets/)
folder. We have taken the original dataset files and transformed them into the
[HuggingFace dataset format][hf-datasets]. All dataset folders contain the
original dataset files, an analysis notebook and a demo file showing how you use
the dataset.

## Experiments

All code for experiments is located in the [experiments](experiments/) folder.
Information on how to reproduce the experiments is available in the
[readme](experiments/README.md) in that folder.

- [BERT](experiments/baseline/) (Devlin et al., 2019)
- [GPT-neo](experiments/template_prompting.py) (Black et al., 2021)
- [(i)PET](experiments/PET/) (Schick and Schütze, 2021)
- [LM-BFF](experiments/AutoPrompt.py) (Gao et al., 2021)
- [Prompt Training with RoBERTa](experiments/ManualPrompt.py) (Liu et al., 2019)

## Results

The results for all methods can be found in the [results](results/) folder.
Information about the results is available in the [readme](results/README.md) in
that folder.

---

## References

Black, S., G. Leo, P. Wang, C. Leahy, and S. Biderman (2021, March). GPT-Neo:
Large scale autoregressive language modelling with mesh-tensorflow.
<https://doi.org/105281/zenodo.5297715>.

Devlin, J., M.-W. Chang, K. Lee, and K. Toutanova (2019, June). BERT: Pre-
training of deep bidirectional transformers for language understanding. In
_Proceedings of the 2019 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies, Volume 1
(Long and Short Papers)_, Minneapolis, Minnesota, pp. 4171–4186. Association for
Computational Linguistics.

Gao, T., A. Fisch, and D. Chen (2021, August). Making pre-trained language
models better few-shot learners. In _Proceedings of the 59th Annual Meeting of
the Association for Computational Linguistics and the 11th International Joint
Conference on Natural Language Processing (Volume 1: Long Papers)_, Online, pp.
3816–3830. Association for Computational Linguistics.

Liu, Y., M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L.
Zettlemoyer, and V. Stoyanov (2019). RoBERTa: A robustly optimized BERT
pretraining approach. _CoRR abs/1907.11692_.

Park, J., & Cardie, C. (2014). Identifying Appropriate Support for Propositions
in Online User Comments. _Proceedings of the First Workshop on Argumentation
Mining_, 29–38. <https://doi.org/10/gg29gq>

Schick, T. and H. Schütze (2021). Exploiting Cloze-Questions for Few-Shot Text
Classification and Natural Language Inference. In _Proceedings of the 16th
Conference of the European Chapter of the Association for Computational
Linguistics: Main Volume_, Online, pp. 255–269. Association for Computational
Linguistics.

<!-- URLs -->

[hf-datasets]: https://huggingface.co/docs/datasets/dataset_script
