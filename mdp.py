import argparse
import os
import pickle
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# List of instruct models we'll use.
from token_prob_experiments import INSTRUCT_MODELS
import sys


class TransitionMatrix:
    def __init__(self, transitions: dict):
        self.transitions = transitions
        self.states = sorted(list(transitions.keys()))
        self.transition_matrix = pd.DataFrame.from_dict(
            transitions, orient="index", columns=self.states
        )

    def __str__(self) -> str:
        """Returns a string of flattened triplets from the transition matrix."""
        triplets = [
            (row, col, self.transition_matrix.loc[row, col])
            for row in self.transition_matrix.index
            for col in self.transition_matrix.columns
        ]
        output_string = ""
        for triplet in triplets:
            output_string += f"({triplet[0]},{triplet[1]},{triplet[2]:.1f}),"
        output_string = output_string[:-1]
        return output_string

    @staticmethod
    def from_transition_matrix(transition_matrix: pd.DataFrame):
        transitions = transition_matrix.to_dict()
        return TransitionMatrix(transitions=transitions)


class Trajectory:
    def __init__(self, possible_states: list, trajectory: str):
        self.possible_states = possible_states
        self.trajectory = trajectory.split()
        self.transition_matrix = self._create_transition_matrix()

    def _create_transition_matrix(self) -> pd.DataFrame:
        current_states = self.trajectory[:-1]
        next_states = self.trajectory[1:]
        all_states = sorted(self.possible_states)

        transition_matrix = pd.crosstab(
            pd.Series(current_states, name="From"),
            pd.Series(next_states, name="To"),
            normalize=1,
        )

        transition_matrix_full = transition_matrix.reindex(
            index=all_states, columns=all_states, fill_value=0
        )
        transition_matrix_full.index.name = None
        transition_matrix_full.columns.name = None

        return transition_matrix_full


# Collate function for dataloader.
def collate_fn(batch):
    # Unpack batch.
    samples = [item["input_ids"] for item in batch]
    samples = torch.IntTensor(samples)
    inputs = {
        "input_ids": samples,
        "attention_mask": torch.ones_like(samples, dtype=torch.int8),
    }

    return inputs


def create_state_token_id_mapping(tokenizer, states: List[str]):
    state_token_id_mapping = {}
    for state in states:
        token = tokenizer.tokenize(str(state))
        token_id = tokenizer.convert_tokens_to_ids(token).pop()
        state_token_id_mapping[state] = token_id
    return state_token_id_mapping


def create_mdp_dataset(
    transition_matrix: dict,
    tokenizer,
    state_token_id_mapping: dict,
    start_state: int,
    num_trajectories: int = 10,
    trajectory_length: int = 25,
):
    """This will generate num_trajectories * (trajectory_length - 1) trajectories of lengths [2, trajectory_length].
    Each set of trajectories of length [2, trajectory_length] continue the same trajectory.

    Args:
        transition_matrix: A dictionary of S keys, each mapping to a list of S values defining transition probabilities.
        tokenizer: The `transformers` tokenizer to use.
        start_state: Which state to start the trajectory from in `transition_matrix`.
        num_trajectories: How many trajectories to subsample from.
        trajectory_length: The maximal length of each trajectory.

    Returns:
        A dataset of trajectories which is a list of dicts containing tokenized trajectory information.
    """
    dataset = []
    states = sorted(list(transition_matrix.keys()))

    for _ in range(num_trajectories):
        current_state = start_state
        trajectory = [start_state]
        for _ in range(trajectory_length - 1):
            seq_data = {}
            state = np.random.choice(states, p=transition_matrix[current_state])
            trajectory.append(int(state))
            current_state = state

        # Convert the trajectory to token IDs
        trajectory = [state_token_id_mapping[str(state)] for state in trajectory]

        # Prepend the <bos> token to the trajectory, making `len(trajectory) == trajectory_length + 1`
        if tokenizer.bos_token_id:
            trajectory.insert(0, tokenizer.bos_token_id)
        else:
            trajectory.insert(0, tokenizer.eos_token_id)
        seq_data = {"input_ids": trajectory, "attention_mask": np.ones_like(trajectory)}
        dataset.append(seq_data)

    return dataset


def generate_input_prompt(trajectory: str):
    """
    Generate input IDs for a model accounting for whether it's an instruct model or not.
    """
    prompt = f"You are given a partial trajectory generated from a Markov process. You continue the trajectory based on your understanding of the Markov process.\n"
    prompt += f"The trajectory is {trajectory}. The next state is: "

    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--quantize", action="store_true", help="Quantize the model.")
    parser.add_argument(
        "--model", type=str, default="google/gemma-2b", help="Model name."
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--in_context_learning", action="store_true", help="Use in-context learning."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing stats."
    )
    parser.add_argument(
        "--num_trials", type=int, default=10, help="Number of trajectories to generate."
    )
    args = parser.parse_args()

    # Maintain some determinism
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config file.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

        # Update args with config.
        for key, value in config.items():
            setattr(args, key, value)

    # Load the tokenizer and model from HuggingFace
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if args.quantize
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        attn_implementation="flash_attention_2",  # TODO add inference-time modification to allow flash-attention.
        torch_dtype=torch.bfloat16,  # NOTE: On unsupported GPUs you'll get emulated bfloat16 support instead of true bfloat16.
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Add padding token if needed.
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.padding_side = "left"

    # Pipeline.
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Determine if using instruct model or not.
    instruct_model = args.model in INSTRUCT_MODELS

    # Path to where we'll store stats.
    results_dir = os.path.join(args.experiment_dir, args.model.replace("/", "_"))
    os.makedirs(results_dir, exist_ok=True)
    stats_name = "stats.pkl" if not args.in_context_learning else "stats_icl.pkl"
    stats_path = os.path.join(results_dir, stats_name)

    # Pre-load stats if we already have some results to avoid recomputing.
    if os.path.exists(stats_path):
        print("Stats have already been calculated! Exiting...")
        sys.exit(0)
    else:
        transitions = []

    trans_matrix = TransitionMatrix(args.transition_matrix)

    state_token_id_mapping = create_state_token_id_mapping(
        tokenizer=tokenizer, states=[str(x) for x in trans_matrix.states]
    )
    state_token_ids = sorted(list(state_token_id_mapping.values()))

    dataset = create_mdp_dataset(
        transition_matrix=args.transition_matrix,
        tokenizer=tokenizer,
        state_token_id_mapping=state_token_id_mapping,
        start_state=args.start_state,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        for b_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Collect logits for all choices.
            all_log_probs = []

            # Put on GPU.
            all_inputs = {
                "input_ids": batch["input_ids"].cuda(),
                "attention_mask": batch["attention_mask"].cuda(),
            }

            outputs = model(
                **all_inputs
            ).logits  # Get the logits out of the model output. [Batch, Sequence, Vocab]

            observed_transitions = outputs[:, 1:, :][:, :, state_token_ids]
            normalized_observed_transitions = F.softmax(observed_transitions, dim=-1)

            for idx in range(all_inputs["input_ids"].shape[0]):
                zipped = list(
                    zip(
                        all_inputs["input_ids"][idx][1:].cpu().detach(),
                        normalized_observed_transitions[idx, :, :].cpu().float().numpy(),
                    )
                )
                transitions.extend(
                    [(tokenizer.convert_ids_to_tokens(a.item()), b) for a, b in zipped]
                )

    pickle.dump(transitions, open(stats_path, "wb"))
