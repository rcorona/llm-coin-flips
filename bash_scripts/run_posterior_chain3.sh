MODELS=(
    google/gemma-2-2b  
    meta-llama/Llama-3.1-8B 
    microsoft/phi-2
)

for MODEL in "${MODELS[@]}"; do
    for ICL_LENGTH in {0..100}; do
        echo -e "\n\nRunning Posterio Chain ICL experiment ICL length: ${ICL_LENGTH}"
        TOKENIZERS_PARALLELISM=true python token_prob_experiments_v2.py configs/exps/contextual/posterior_chain_coin_flip/icl_${ICL_LENGTH}/bias_50.yaml ${MODEL};
        wait; 
    done
done