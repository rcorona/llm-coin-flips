import os
import glob
import yaml
import torch
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

config_file_paths = glob.glob(
    "/home/ericwang0533/random_needles/random_needles/configs/exps/contextual/die_roll_icl_v2/icl_100/bias_*.yaml"
)

model_names = [
    # "google/gemma-2-2b-it",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "microsoft/Phi-3.5-mini-instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "allenai/OLMoE-1B-7B-0924-Instruct",
    # "google/gemma-2-2b",
    "meta-llama/Llama-3.1-8B",
    # "microsoft/phi-2",
    # "mistralai/Mistral-7B-v0.3",
    # "allenai/OLMoE-1B-7B-0924",
]

device = torch.device("cpu")  # or "cpu", or check cuda availability

# Define a top-level directory for your results
RESULTS_ROOT = "/home/ericwang0533/random_needles/random_needles/attention_weights/results_incontext_die_roll"

for model_name in model_names:
    try:
        # Sanitize model name: replace slashes with underscores
        sanitized_model_name = model_name.replace("/", "_")

        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(device)

        for config_path in config_file_paths:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Change use/not instruction
            bias_prompts = config.get("bias_needles", [])
            input_prompts = config.get("input_prompt", [])
            # N = 1  # number of in-context examples
            selected_incontext = input_prompts[0]

            final_prompts = []
            for bias_text in bias_prompts:
                # final_prompt = f"{bias_text} {selected_incontext} "
                final_prompt = f"{selected_incontext} "
                final_prompts.append(final_prompt)

            # Extract the subdir name ("icl_5" or "icl_1", etc.)
            # and the base file name ("bias_0" etc.)
            config_dir = os.path.dirname(config_path)        # e.g. /home/.../coin_flip_icl_v2/icl_5
            config_subdir = os.path.basename(config_dir)     # e.g. icl_5
            config_base = os.path.splitext(os.path.basename(config_path))[0]  # e.g. bias_0

            # Construct the output directory
            out_dir_base = os.path.join(
                RESULTS_ROOT,
                sanitized_model_name,  # e.g. google_gemma-2-2b-it
                config_subdir,         # e.g. icl_5
                config_base            # e.g. bias_0
            )
            os.makedirs(out_dir_base, exist_ok=True)

            for idx, prompt in enumerate(final_prompts):
                print(
                    f"\n[MODEL: {model_name}] "
                    f"[CONFIG: {os.path.basename(config_path)}] "
                    f"Prompt {idx+1}/{len(final_prompts)}"
                )
                print("Prompt:\n", prompt)

                model_inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

                output = model.generate(
                    model_inputs,
                    output_attentions=True,
                    max_new_tokens=1,
                    return_dict_in_generate=True
                )

                attentions_list = output.attentions[-1]

                max_head_attentions = []
                for layer_attention in attentions_list:
                    # layer_attention: (batch_size=1, num_heads, seq_len_q, seq_len_kv)
                    max_over_heads, _ = layer_attention.max(dim=1)  # => (1, seq_len_q, seq_len_kv)
                    max_head_attentions.append(max_over_heads)

                # Stack across layers => shape: (num_layers, 1, seq_len_q, seq_len_kv)
                stacked_attentions = torch.stack(max_head_attentions, dim=0)
                # Max over layers => shape: (1, seq_len_q, seq_len_kv)
                max_over_layers, _ = stacked_attentions.max(dim=0)

                max_attention = max_over_layers[0]  # => shape: (seq_len_q, seq_len_kv)

                seq_len_q, seq_len_kv = max_attention.shape
                last_token_idx = seq_len_q - 1

                num_input_tokens = model_inputs.shape[1]
                attention_to_input = max_attention[last_token_idx, :num_input_tokens]

                input_tokens = tokenizer.convert_ids_to_tokens(model_inputs[0])
                attention_values = attention_to_input.detach().cpu().numpy()

                # Plot
                # plt.figure(figsize=(10, 4))
                n_tokens = len(input_tokens)
                plt.figure(figsize=(n_tokens * 0.25, 5), dpi=200)  
                sns.barplot(x=input_tokens, y=attention_values)
                plt.xticks(rotation=90)
                plt.xlabel("Input Tokens")
                plt.ylabel("Attention Weight")
                plt.title(
                    f"Max Attention (Gen Token -> Input Tokens)\n"
                    f"{model_name}\nConfig: {os.path.basename(config_path)} | Prompt {idx+1}"
                )
                plt.tight_layout()

                # Save figure
                fig_name = f"prompt_{idx+1}.png"
                fig_path = os.path.join(out_dir_base, fig_name)
                plt.savefig(fig_path, dpi=300)
                plt.close()

                # Save numerical results
                csv_name = f"prompt_{idx+1}.csv"
                csv_path = os.path.join(out_dir_base, csv_name)
                with open(csv_path, mode="w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["token", "attention"])
                    for t, val in zip(input_tokens, attention_values):
                        writer.writerow([t, val])

        del model
        del tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
