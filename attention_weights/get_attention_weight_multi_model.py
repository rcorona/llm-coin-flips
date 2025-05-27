from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Define lists of models and prompts
model_names = [
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMoE-1B-7B-0924-Instruct",
    "google/gemma-2-2b",
    "meta-llama/Llama-3.1-8B",
    "microsoft/phi-2",
    "mistralai/Mistral-7B-v0.3",
    "allenai/OLMoE-1B-7B-0924",
]

prompts = [
    "When I flip coins they land on heads 0% of the time. I flipped a coin and it landed on ",
    "When I flip coins they land on heads 10% of the time. I flipped a coin and it landed on ",
    "When I flip coins they land on heads 20% of the time. I flipped a coin and it landed on ",
    "When I flip coins they land on heads 30% of the time. I flipped a coin and it landed on ",
    "When I flip coins they land on heads 40% of the time. I flipped a coin and it landed on ",
    "When I flip coins they land on heads 50% of the time. I flipped a coin and it landed on ",
    "When I flip coins they land on heads 60% of the time. I flipped a coin and it landed on ",
    "When I flip coins they land on heads 70% of the time. I flipped a coin and it landed on ",
    "When I flip coins they land on heads 80% of the time. I flipped a coin and it landed on ",
    "When I flip coins they land on heads 90% of the time. I flipped a coin and it landed on ",
    "When I flip coins they land on heads 100% of the time. I flipped a coin and it landed on ",
]

# Set up device
device = torch.device('cpu')

for model_name in model_names:
    print(f"Loading model: {model_name}")
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    
    for idx, prompt in enumerate(prompts):
        print(f"Processing prompt {idx+1}/{len(prompts)}: {prompt}")
        # Tokenize and encode the prompt
        model_inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate outputs with attentions
        output = model.generate(
            model_inputs,
            output_attentions=True,
            max_new_tokens=1,
            return_dict_in_generate=True
        )
        print(output.attentions[0][0].shape)
        
        # Extract attentions for the first generated token
        attentions_for_token = output.attentions[0]  # List of tensors for each layer
        
        # Compute the maximum over heads and layers
        max_head_attentions = []
        for layer_attention in attentions_for_token:
            max_head_attention, _ = layer_attention.max(dim=1)  # Max over heads
            max_head_attentions.append(max_head_attention)
        
        stacked_max_head_attentions = torch.stack(max_head_attentions)
        max_attention_over_layers, _ = stacked_max_head_attentions.max(dim=0)  # Max over layers
        
        # Prepare tokens for visualization
        input_tokens = tokenizer.convert_ids_to_tokens(model_inputs[0])
        generated_tokens = tokenizer.convert_ids_to_tokens(output.sequences[0])
        # max_attention_over_layers has shape [1, seq_len_q, seq_len_kv]
        seq_len_q, seq_len_kv = max_attention_over_layers.shape[1], max_attention_over_layers.shape[2]

        # Index of the last token in the query sequence (the generated token)
        last_token_idx = seq_len_q - 1

        # Number of input tokens
        num_input_tokens = model_inputs.shape[1]

        # Extract attention from the generated token to input tokens
        attention_to_input = max_attention_over_layers[0, last_token_idx, :num_input_tokens]
        print(attention_to_input.shape)

        # Tokens corresponding to input
        tokens = input_tokens

        # Convert to numpy
        attention_values = attention_to_input.detach().cpu().numpy()

        # Plot
        plt.figure(figsize=(10, 4))
        # sns.barplot(x=tokens, y=attention_values)
        plt.bar(tokens, attention_values)
        plt.xticks(rotation=90)
        plt.xlabel('Input Tokens')
        plt.ylabel('Attention Weight')
        plt.title(f'Maximum Attention from Generated Token to Input Tokens\nModel: {model_name}\nPrompt {idx+1}')
        plt.tight_layout()
        file_name = f'./results_incontext/{model_name}_prompt{idx+1}.png'.replace("/", "_")
        plt.savefig(file_name, dpi=300)
        plt.close()
        
    # Free up memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
