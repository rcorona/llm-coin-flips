from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_inputs = tokenizer.encode("When I flip coins they land on heads 10% of the time. I flipped a coin and it landed on", return_tensors="pt").to(model.device)

output = model.generate(model_inputs, output_attentions=True, max_new_tokens=5, return_dict_in_generate=True)


# Get attentions for the first generated token
attentions_for_token = output.attentions[0]  # List of tensors for each layer

# Initialize a list to store max attention per layer
max_head_attentions = []

# Iterate over layers
for layer_attention in attentions_for_token:
    # layer_attention shape: [1, num_heads, seq_len, seq_len]
    
    # Compute the max over heads
    max_head_attention, _ = layer_attention.max(dim=1)  # Shape: [1, seq_len, seq_len]
    
    # Append to the list
    max_head_attentions.append(max_head_attention)

# Stack the max_head_attentions tensors
stacked_max_head_attentions = torch.stack(max_head_attentions)  # Shape: [num_layers, 1, seq_len, seq_len]

# Compute the max over layers
max_attention_over_layers, _ = stacked_max_head_attentions.max(dim=0)  # Shape: [1, seq_len, seq_len]

# Original input tokens
input_tokens = tokenizer.convert_ids_to_tokens(model_inputs[0])

# Include generated tokens up to the current step
generated_tokens = tokenizer.convert_ids_to_tokens(output.sequences[0])

# Ensure tokens align with the attention matrix
seq_len = max_attention_over_layers.shape[-1]
tokens = generated_tokens[:seq_len]

import matplotlib.pyplot as plt
import seaborn as sns

# Convert attention to numpy array and remove batch dimension
attention_matrix = max_attention_over_layers[0].detach().cpu().numpy()

# Plot the heatmap
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
