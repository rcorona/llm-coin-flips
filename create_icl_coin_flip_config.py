import yaml
import random
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


def generate_configs(args):
    # Define the parameters
    thetas = list(range(args.theta_start, args.theta_end + args.theta_step, args.theta_step))
    print(f"thetas = {thetas}")
    choices = args.choices

    # List of ICL lengths
    icl_values = list(range(args.icl_start, args.icl_end, args.icl_step))
    print(f"ICL = {icl_values}")

    # Base experiment directory
    base_experiment_dir = Path(args.experiment_base_dir)

    instruct_prompt = [
        "Please complete this sentence: I just flipped a coin, and it landed on ",
        "Finish this sentence: The result of my coin flip is ",
        "Complete the sentence: After flipping the coin, it landed on ",
        "Fill in the rest: I flipped a coin, and the outcome was ",
        "Complete the phrase: The coin flip resulted in ",
    ]

    input_prompts = [
        "I just flipped a coin, and it came up ",
        "The result of my coin flip is ",
        "After flipping the coin, it landed on ",
        "I flipped a coin, and the outcome was ",
        "The coin flip resulted in ",
        "After tossing the coin, it ended up on ",
        "I tossed a coin, and it fell on ",
        "The coin I flipped landed on ",
        "The coin I tossed resulted in ",
        "Following the coin flip, it showed ",
        "The coin fell on ",
        "The flip of the coin resulted in ",
        "I flipped the coin, and it settled on ",
        "The result after flipping the coin is ",
        "The outcome of my coin flip is ",
        "I tossed the coin, and the outcome is ",
        "The result of my coin toss is ",
        "I flipped the coin, and it came up ",
        "The coin came to rest on ",
        "After flipping, the coin showed ",
        "The toss of the coin revealed ",
        "I flipped the coin, and it turned up ",
        "The coin toss ended with ",
        "After tossing the coin, it showed ",
        "The coin flipped over to ",
        "After flipping, the coin settled on ",
        "My coin toss resulted in ",
        "The outcome of my coin flip turned out to be ",
        "I flipped the coin, and its final position was ",
        "The coin fell, showing ",
        "I tossed the coin, and it landed showing ",
        "Following the toss, the coin showed ",
        "The flip resulted in the coin landing on ",
        "The coin toss revealed ",
        "The outcome of the coin landing is ",
        "After tossing, the coin landed on ",
        "I flipped the coin and saw it land on ",
        "After the flip, the coin showed ",
        "The result of tossing the coin was ",
        "When I flipped the coin, it landed on ",
        "The coin showed this side after the flip: ",
        "The flip of the coin ended with ",
        "After tossing, the coin fell to show ",
        "The result of my toss came out as ",
        "The toss of the coin came to rest on ",
        "The coin after the flip landed on ",
        "I flipped the coin, and it ended on ",
        "The result of the coin toss ended up being ",
        "I flipped a coin, and its final side was ",
        "The coin flip showed the result: ",
    ]

    # For reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Now we will loop over the thetas list
    for theta in (pbar := tqdm(thetas)):
        pbar.set_description_str(f"Theta = {theta}")

        data = {}
        data["needle_expected_dists"] = []
        data["instruct_bias_needles"] = []
        data["bias_needles"] = []
        data["choices"] = choices

        # Create bias needle and instruct bias needle
        if args.bias_needle:
            bias_needle = f"When I flip coins they land on heads {theta}% of the time."
            instruct_bias_needle = f"When you flip coins, they land on heads {theta}% of the time."
            data['bias_needles'].append(bias_needle)
            data['instruct_bias_needles'].append(instruct_bias_needle)
        else:
            data['bias_needles'] = ""
            data['instruct_bias_needles'] = ""

        # Calculate expected distributions
        heads_prob = theta / 100.0
        tails_prob = round(1.0 - heads_prob, 1)
        data["needle_expected_dists"].append([heads_prob, tails_prob])

        # Generate the max ICL length possible
        flips = np.random.choice(choices, icl_values[-1], p=[heads_prob, tails_prob])

        # Now we will generate separate YAML files for each icl_value
        for icl_value in icl_values:
            # Create output directory for this icl_examples value
            output_dir = base_experiment_dir / f"icl_{icl_value}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Adjust the experiment_dir to include icl_examples
            data["experiment_dir"] = str(
                Path(*base_experiment_dir.parts[1:]) / f"icl_{icl_value}" / f"theta_{theta}"
            )

            data["instruct_prompt"] = instruct_prompt
            data["input_prompt"] = []

            # Now generate the adjusted input prompts
            adjusted_input_prompts = []

            for prompt in input_prompts:
                icl_examples_text = ", then ".join(flips[:icl_value])
                if icl_value == 0:
                    adjusted_prompt = f"{prompt}{icl_examples_text}"
                else:
                    adjusted_prompt = f"{prompt}{icl_examples_text}, then "
                adjusted_input_prompts.append(adjusted_prompt)

            # Assign the adjusted input prompts to the data
            data["input_prompt"] = adjusted_input_prompts

            # Write the data to a YAML file
            filename = f"theta_{theta}.yaml"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                yaml.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create posterior chain coin flip configs",
    )

    parser.add_argument(
        "--theta-start",
        type=int,
        help="The starting range for the thetas to sweep over",
        default=0,
    )
    parser.add_argument(
        "--theta-end",
        type=int,
        help="The end range for the thetas to sweep over",
        default=110,
    )
    parser.add_argument(
        "--theta-step",
        type=int,
        help="The step size for the list of thetas to sweep over",
        default=10,
    )
    parser.add_argument(
        "--choices",
        nargs="+",
        help="The available choices (the code supports two right now)",
        default=["heads", "tails"],
    )
    parser.add_argument(
        "--icl-start",
        type=int,
        help="The starting range for the ICL length to sweep over",
        required=True,
        default=0,
    )
    parser.add_argument(
        "--icl-end",
        type=int,
        help="The starting range for the ICL length to sweep over",
        required=True,
        default=100,
    )
    parser.add_argument(
        "--icl-step",
        type=int,
        help="The step size for the list of ICLs to sweep over",
        required=True,
        default=10,
    )
    parser.add_argument(
        "--bias-needle",
        action="store_true",
        help="Whether to include a bias needle or not"
    )
    parser.add_argument(
        "--experiment-base-dir",
        type=str,
        help="Where to save the config files",
        required=True,
    )
    parser.add_argument("--seed", type=int, help="Seed for reproducibility", default=42)

    args = parser.parse_args()
    generate_configs(args)
