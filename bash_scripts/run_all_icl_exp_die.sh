#!/bin/bash

# List of all models being used.
INSTRUCT_MODELS=(
    google/gemma-2-2b-it
    meta-llama/Llama-3.1-8B-Instruct
    microsoft/Phi-3.5-mini-instruct
    mistralai/Mistral-7B-Instruct-v0.3
    allenai/OLMoE-1B-7B-0924-Instruct
)

STD_MODELS=(
    google/gemma-2-2b
    meta-llama/Llama-3.1-8B
    microsoft/phi-2
    mistralai/Mistral-7B-v0.3
    allenai/OLMoE-1B-7B-0924
)

# Select subset of models to run based on arguments.
MODEL_TYPE=$1

ICL_LENGTHS=(1 3 5 10 20 100)
BIASES=(0 20 40 60 80 100)
BIAS_TARGETS=(1 2 3 4 5 6)

# Select the appropriate models based on MODEL_TYPE.
if [ "$MODEL_TYPE" == "instruct" ]; then
    MODELS=("${INSTRUCT_MODELS[@]}")
elif [ "$MODEL_TYPE" == "std" ]; then
    MODELS=("${STD_MODELS[@]}")
else
    MODELS=("${INSTRUCT_MODELS[@]}" "${STD_MODELS[@]}")
fi

# List of GPUs to use.
GPUS=(0 1 2 3 4 5 6 7)

# Map models to GPUs using CUDA_VISIBLE_DEVICES.
# If there are more models than GPUs, assign multiple models per GPU.
declare -A GPU_MODEL_QUEUE

# Initialize queues for each GPU.
for GPU in "${GPUS[@]}"; do
    GPU_MODEL_QUEUE["$GPU"]=""
done

GPU_COUNT=${#GPUS[@]}

# Assign models to GPUs.
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    GPU_INDEX=$(( i % GPU_COUNT ))
    GPU=${GPUS[$GPU_INDEX]}
    GPU_MODEL_QUEUE["$GPU"]+="$MODEL;"
done

# Now, for each GPU, run experiments for assigned models sequentially.
for GPU in "${GPUS[@]}"; do
    (
        export CUDA_VISIBLE_DEVICES=$GPU
        MODELS_ON_GPU="${GPU_MODEL_QUEUE["$GPU"]}"
        IFS=';' read -r -a MODELS_ARRAY <<< "$MODELS_ON_GPU"
        for MODEL in "${MODELS_ARRAY[@]}"; do
            if [ -n "$MODEL" ]; then
                echo "Running experiments for model $MODEL on GPU $GPU"
                for ICL_LENGTH in "${ICL_LENGTHS[@]}"; do
                    for BIAS_TARGET in "${BIAS_TARGETS[@]}"; do
                        for BIAS in "${BIASES[@]}"; do
                            echo -e "\n\nRunning icl experiment for model: ${MODEL} with icl_length: ${ICL_LENGTH}, bias_target: ${BIAS_TARGET}, and bias: ${BIAS}"
                            bash bash_scripts/run_token_prob_exp_v2.sh "${MODEL}" "die_roll_icl_v2/icl_${ICL_LENGTH}/bias_${BIAS_TARGET}_${BIAS}"
                        done
                    done
                done
            fi
        done
    ) &
done
wait
