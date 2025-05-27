# LLM Coin Flips & Random Processes

[![Paper](https://img.shields.io/badge/arXiv-2402.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2402.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Project Website](https://img.shields.io/badge/project-website-blue)](https://ai-climate.berkeley.edu/llm-coin-flips/)

## Overview

**LLM Coin Flips** is a research framework and codebase for investigating how large language models (LLMs) simulate random processes—such as biased coin flips, die rolls, and Poisson processes—using *in-context learning* (ICL). Our experiments reveal that, when given enough in-context examples, LLMs update their beliefs in a manner closely matching Bayesian inference, despite starting with biased priors.

This repository accompanies the paper:  
**"Enough Coin Flips Can Make LLMs Act Bayesian"**  
*Ritwik Gupta, Rodolfo Corona, Jiaxin Ge, Eric Wang, Dan Klein, Trevor Darrell, David M. Chan*  
[Read the paper](https://arxiv.org/abs/2402.XXXXX) | [Project page & visualizations](https://ai-climate.berkeley.edu/llm-coin-flips/)

---

## Key Findings

- **LLMs show biased priors** for stochastic processes (e.g., preferring "heads" in coin flips).
- **In-context learning (ICL)**: Supplying sequences of actual outcomes allows LLMs to *approximately* update beliefs in a Bayesian fashion.
- **Bayesian behavior**: With sufficient demonstrations, LLMs’ posterior predictions closely match normative Bayesian inference; deviations are mainly due to miscalibrated priors, not flawed updates.
- **Attention**: The magnitude of attention in transformer models has minimal effect on the quality of Bayesian inference in these tasks.
- **Instruction tuning**: Instruct-tuned models incorporate in-context evidence differently, often with a shorter time horizon for updates.

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Create Configuration Files

`python create_configs.py configs/exps/contextual/ exps/contextual/`

This will create the files `coin_flip.yaml`, `die_roll.yaml`, and `random_number.yaml` in
the `configs/exps/contextual` folder.

## Run Token Probability Experiments

### For single experiment on a model:

`bash bash_scripts/run_token_prob_exp.sh $HF_MODEL_NAME $EXP_TYPE [--overwrite]`

Here, `$EXP_TYPE` is one of `coin_flip`, `die_roll`, or `randon_number`.
The `--overwrite` flag can be optionally added to overwrite any results which have already been collected.

### For running multiple experiments in sequence:

`bash bash_scripts/run_all_token_prob_exp.sh $MODEL_TYPE $EXP_TYPE [--overwrite]`
Here, `$MODEL_TYPE` is one of `std` or `instruct`.
The script will loop over all models of each of those types (std being non-instruct models) to run the desired token prob experiments.

## Check Token Probability Experiment Progress

`python check_exp_progres.py configs/exps/contextual/$CFG_NAME.yaml --experiment_dir exps/contextual/$CFG_NAME`

For example, for the random number experiments one would run:

```
python check_exp_progress.py configs/exps/contextual/die_roll.yaml --experiment_dir exps/contextual/die_roll/
```

## Poisson Experiments

To run all experiments (using similar nomenclature as above):

```
bash bash_scripts/run_all_poisson_exp.sh $MODEL_TYPE $EXP_TYPE [--overwrite]
```

Here `$EXP_TYPE` is one of `std_exp` or `icl`, denoting running the standard poisson experiments with 
explicit prompting (std_exp, e.g. generate a story with lambda=0.6) or in-context-learning (icl, where the model is prompted with a pre-generate prefix 
containing the special character with the correct lambda proportion and is meant to generate new text with the same statistics of character insertion). 

To run a single experiment: 

```
bash bash_scripts/run_poisson_exp.sh $MODEL $EXP_TYPE [--overwrite]
```

## Links

- [Project Website & Visualizations](https://ai-climate.berkeley.edu/llm-coin-flips/)
- [Paper on arXiv](https://arxiv.org/abs/2402.XXXXX)
- [Code License (MIT)](LICENSE)

---

## Acknowledgments

This project was supported by the Berkeley AI Research (BAIR) Lab and affiliated grants.  
Special thanks to the contributors and reviewers!

---

**Questions?**  
Open an issue or reach out via the project page.
