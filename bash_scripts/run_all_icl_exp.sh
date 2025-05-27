#!/bin/bash

# List of all models being used.
INSTRUCT_MODELS=(
    # google/gemma-2-2b-it
    # meta-llama/Llama-3.1-8B-Instruct
    microsoft/Phi-3.5-mini-instruct
    # mistralai/Mistral-7B-Instruct-v0.3
    # allenai/OLMoE-1B-7B-0924-Instruct
)

STD_MODELS=(
    # google/gemma-2-2b # TODO
    # meta-llama/Llama-3.1-8B # TODO
    microsoft/phi-2
    # mistralai/Mistral-7B-v0.3
    # allenai/OLMoE-1B-7B-0924
)

# Select subset of models to run based on arguments.
MODEL_TYPE=$1

ICL_LENGTHS=(1 3 5 10 20 100)
BIASES=(0 10 20 30 40 50 60 70 80 90 100)

# Select the appropriate models based on MODEL_TYPE.
if [ "$MODEL_TYPE" == "instruct" ]; then
    MODELS=("${INSTRUCT_MODELS[@]}")
elif [ "$MODEL_TYPE" == "std" ]; then
    MODELS=("${STD_MODELS[@]}")
else
    MODELS=("${INSTRUCT_MODELS[@]}" "${STD_MODELS[@]}")
fi

for MODEL in "${MODELS[@]}"; do
    for ICL_LENGTH in "${ICL_LENGTHS[@]}"; do
        for BIAS in "${BIASES[@]}"; do
            echo -e "\n\nRunning icl experiment for model: ${MODEL} with icl_length: ${ICL_LENGTH} and bias: ${BIAS}";
            bash bash_scripts/run_token_prob_exp_v2.sh ${MODEL} coin_flip_icl_v2/icl_${ICL_LENGTH}/bias_${BIAS};
            wait;
        done
    done
done
