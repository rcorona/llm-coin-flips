STD_MODELS=(
    # google/gemma-2-2b
    # meta-llama/Llama-3.1-8B
    # microsoft/phi-2
    # mistralai/Mistral-7B-v0.3
    allenai/OLMoE-1B-7B-0924
)

# Config file
EXP_TYPE=$1

for MODEL in "${STD_MODELS[@]}"; do
    echo -e "\n\nRunning ${EXP_TYPE} experiment for model: ${MODEL}";
    CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true python mdp.py configs/exps/mdp/${EXP_TYPE}.yaml --model ${MODEL} --batch_size 16;
    wait;
done