# List of all models being used.
INSTRUCT_MODELS=(
    google/gemma-2-2b-it
    meta-llama/Llama-3.1-8B-Instruct
    microsoft/Phi-3.5-mini-instruct
    mistralai/Mistral-7B-Instruct-v0.3
    allenai/OLMoE-1B-7B-0924-Instruct
)

STD_MODELS=(
    google/gemma-2-2b # TODO
    meta-llama/Llama-3.1-8B # TODO
    microsoft/phi-2
    mistralai/Mistral-7B-v0.3
    allenai/OLMoE-1B-7B-0924
)

# Select subset of models to run based on arguments.
MODEL_TYPE=$1

CONFIG_BASE_FOLDER=$2

# Determine whether to overwrite the existing results.
CONFIGS=($(find ${CONFIG_BASE_FOLDER} -type f -name "*.yaml"))

# Determine if we should overwrite the existing results.
if [ $# -eq 3 ]; then
    OVERWRITE=$3
else
    OVERWRITE=""
fi

if [ $MODEL_TYPE == "instruct" ]; then
    MODELS=("${INSTRUCT_MODELS[@]}")
elif [ $MODEL_TYPE == "std" ]; then
    MODELS=("${STD_MODELS[@]}")
else
    MODELS=("${INSTRUCT_MODELS[@]}" "${STD_MODELS[@]}")
fi

for CONFIG in "${CONFIGS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo -e "\n\nRunning ${CONFIG} experiment for model: ${MODEL}";
        bash bash_scripts/run_token_prob_exp_v2.sh ${CONFIG} ${MODEL} ${OVERWRITE};
        wait;
    done
done
