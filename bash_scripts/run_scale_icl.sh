# List of all model sizes being used.
SIZES=(
#    70m
    160m 
    410m
    1b
    1.4b
    2.8b
    6.9b
    12b
)

ICL_LENGTHS=(
    1
    3
    5
    10
    20
    100
)

BIASES=(
    0
    10
    20
    30
    40
    50
    60
    70
    80
    90
    100
)

# Determine if we should overwrite the existing results.
if [ $# -eq 3 ]; then
    OVERWRITE=$1
else
    OVERWRITE=""
fi

for SIZE in "${SIZES[@]}"; do 
    for ICL_LENGTH in "${ICL_LENGTHS[@]}"; do 
        for BIAS in "${BIASES[@]}"; do 
            echo -e "\n\nRunning ICL experiment for model size: ${SIZE}, ICL length: ${ICL_LENGTH}, and bias: ${BIAS}";
            TOKENIZERS_PARALLELISM=true python token_prob_experiments_v2.py configs/exps/contextual/coin_flip_icl_v2/icl_${ICL_LENGTH}/bias_${BIAS}.yaml EleutherAI/pythia-${SIZE}-deduped ${OVERWRITE};
            wait; 
        done
    done
done