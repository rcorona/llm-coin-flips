from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_inputs = tokenizer.encode("When I flip coins they land on heads 10% of the time. I flipped a coin and it landed on", return_tensors="pt").to(model.device)

output = model.generate(model_inputs, output_attentions=True, max_new_tokens=5, return_dict_in_generate=True)


# Step 1: Extract attentions for the first generated token
attentions_for_token = output.attentions[0]  # List of tensors for each layer

# Step 2: Compute the maximum over heads for each layer
max_head_attentions = []
for layer_attention in attentions_for_token:
    # layer_attention shape: [1, num_heads, seq_len, seq_len]
    max_head_attention, _ = layer_attention.max(dim=1)  # Shape: [1, seq_len, seq_len]
    max_head_attentions.append(max_head_attention)

# Step 3: Compute the maximum over layers
stacked_max_head_attentions = torch.stack(max_head_attentions)  # Shape: [num_layers, 1, seq_len, seq_len]
max_attention_over_layers, _ = stacked_max_head_attentions.max(dim=0)  # Shape: [1, seq_len, seq_len]

# Step 4: Prepare tokens
input_tokens = tokenizer.convert_ids_to_tokens(model_inputs[0])
generated_tokens = tokenizer.convert_ids_to_tokens(output.sequences[0])
seq_len = max_attention_over_layers.shape[-1]
tokens = generated_tokens[:seq_len]

# Step 5: Visualize the attention matrix
attention_matrix = max_attention_over_layers[0].detach().cpu().numpy()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(
    attention_matrix,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap='viridis'
)
plt.xlabel('Key Tokens')
plt.ylabel('Query Tokens')
plt.title('Maximum Attention Weights Over Heads and Layers')
plt.savefig('max_attention_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Visualize attention from the generated token to input tokens
# Index of the last token (the generated token)
last_token_idx = seq_len - 1

# Number of input tokens
num_input_tokens = model_inputs.shape[1]

# Extract attention from the generated token to input tokens
attention_to_input = max_attention_over_layers[0, last_token_idx, :num_input_tokens]
tokens = generated_tokens[:num_input_tokens]

# Convert to numpy
attention_values = attention_to_input.detach().cpu().numpy()

# Plot
plt.figure(figsize=(10, 4))
sns.barplot(x=tokens, y=attention_values)
plt.xticks(rotation=90)
plt.xlabel('Input Tokens')
plt.ylabel('Attention Weight')
plt.title('Maximum Attention from Generated Token to Input Tokens')
plt.tight_layout()
plt.savefig('max_attention_barplot.png', dpi=300)
plt.show()
