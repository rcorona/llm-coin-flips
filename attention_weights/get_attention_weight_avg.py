from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_inputs = tokenizer.encode("When I flip coins they land on heads 10% of the time. I flipped a coin and it landed on", return_tensors="pt").to(model.device)

output = model.generate(model_inputs, output_attentions=True, max_new_tokens=5, return_dict_in_generate=True)


# Get attentions for the first generated token
attentions_for_token = output.attentions[0]  # List of tensors for each layer

import torch

# Initialize a tensor to accumulate attention weights
attention_accumulator = None

# Iterate over layers
for layer_attention in attentions_for_token:
    # layer_attention shape: [1, num_heads, seq_len, seq_len]
    
    # Average over heads
    avg_head_attention = layer_attention.mean(dim=1)  # Shape: [1, seq_len, seq_len]
    
    # Accumulate over layers
    if attention_accumulator is None:
        attention_accumulator = avg_head_attention
    else:
        attention_accumulator += avg_head_attention

# Average over layers
avg_attention = attention_accumulator / len(attentions_for_token)  # Shape: [1, seq_len, seq_len]

# Original input tokens
input_tokens = tokenizer.convert_ids_to_tokens(model_inputs[0])

# Include generated tokens up to the current step
generated_tokens = tokenizer.convert_ids_to_tokens(output.sequences[0])

# Since the sequence length might be fixed, ensure tokens align with attention matrices
seq_len = avg_attention.shape[-1]
tokens = generated_tokens[:seq_len]

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Convert attention to numpy array and remove batch dimension
attention_matrix = avg_attention[0].detach().cpu().numpy()

# Create a mask to hide padding tokens if necessary (not shown here)

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
plt.title('Attention Weights Averaged Over Heads and Layers')
# Save the figure to a file
plt.savefig('attention_heatmap.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
