# Random Needles

## Installation

```
pip install -r requirements.txt
spacy download en_core_web_sm
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