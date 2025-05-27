# Arguments. 
MODEL=$1

# Determine whether to overwrite the existing results.
EXP_TYPE=$2

if [ $EXP_TYPE == "icl" ]; then
    EXP_TYPE="--in_context_learning"
else
    EXP_TYPE=""
fi

# Check if 3 arguments were provided. 
if [ $# -eq 3 ]; then
    OVERWRITE=$3
else
    OVERWRITE=""
fi

# Coin flip experiment with context, will generate a pivot table. TODO perform both contextual and non-contextual experiments at the same time. 
TOKENIZERS_PARALLELISM=true python poisson.py --model ${MODEL} ${EXP_TYPE} ${OVERWRITE}