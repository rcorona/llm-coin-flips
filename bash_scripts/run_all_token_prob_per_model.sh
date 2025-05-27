# Select subset of models to run based on arguments. 
MODEL_NAME=$1

# Determine if we should overwrite the existing results.
if [ $# -eq 2 ]; then
    OVERWRITE=$2
else
    OVERWRITE=""
fi

EXPS=(
    "coin_flip"
    "die_roll"
    "random_number"
)

for EXP in "${EXPS[@]}"; do 
    echo -e "\n\nRunning ${EXP} experiment for model: ${MODEL_NAME}";
    bash bash_scripts/run_token_prob_exp.sh ${MODEL_NAME} ${EXP} ${OVERWRITE};
    wait; 
done