CONFIG=$1

# Arguments.
MODEL=$2

# Check if 3 arguments were provided.
if [ $# -eq 3 ]; then
    OVERWRITE=$3
else
    OVERWRITE=""
fi

# Coin flip experiment with context, will generate a pivot table. TODO perform both contextual and non-contextual experiments at the same time.
TOKENIZERS_PARALLELISM=true python token_prob_experiments_v2.py ${CONFIG} ${MODEL} ${OVERWRITE};
