import yaml
import random
import os
from tqdm import tqdm
import argparse
from pathlib import Path


def generate_configs(args):
    # Define the parameters
    bias_percentages = range(args.theta_start, args.theta_end, args.theta_step)
    coin_p1 = args.theta_one
    coin_p2 = args.theta_two
    cutoff = args.cutoff
    assert args.icl_start <= cutoff <= args.icl_end, "The cutoff value must be in [icl_start, icl_end]"

    choices = args.choices

    context_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1000]  # Unneeded?

    default_expected_dist = [0.5, 0.5]  # Unneeded?

    document_depths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Unneeded?

    # List of icl_examples values
    icl_examples_list = list(range(args.icl_start, args.icl_end))

    # Base experiment directory
    base_experiment_dir = Path(args.experiment_base_dir)  # 'configs/exps/contextual/posterior_chain_coin_flip'

    instruct_prompt = [
        "Please complete this sentence: I just flipped a coin, and it landed on ",
        "Finish this sentence: The result of my coin flip is ",
        "Complete the sentence: After flipping the coin, it landed on ",
        "Fill in the rest: I flipped a coin, and the outcome was ",
        "Complete the phrase: The coin flip resulted in "
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
        "The coin flip showed the result: "
    ]

    # For reproducibility
    random.seed(args.seed)

    # Generate icl_examples number of coin flips according to the bias probability
    icl_flips = []

    # Top loop is ICL length so that we have the same sequence of coin flips, progressively growing over ICL lengths. 
    for icl_length in tqdm(icl_examples_list):
        
        # Create output directory for this icl_examples value
        output_dir = base_experiment_dir / f'icl_{icl_length}'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Now we will generate separate YAML files for each bias percentage pair. 
        for bias_percentage in bias_percentages:
            # Create data dictionary for this bias percentage and icl_examples
            data = {}
            data['bias_needles'] = []
            data['choices'] = choices
            data['context_lengths'] = context_lengths
            data['default_expected_dist'] = default_expected_dist
            data['document_depths'] = document_depths

            # Adjust the experiment_dir to include icl_examples
            data['experiment_dir'] = os.path.join(base_experiment_dir, f'icl_{icl_length}', f'{bias_percentage}')

            data['instruct_prompt'] = instruct_prompt
            data['input_prompt'] = []
            data['instruct_bias_needles'] = []
            data['needle_expected_dists'] = []

            # Create bias needle and instruct bias needle
            bias_needle = f""
            instruct_bias_needle = f""
            data['bias_needles'].append(bias_needle)
            data['instruct_bias_needles'].append(instruct_bias_needle)

            # Add P1 and P2 to the data, as well as cutoff. 
            data['P1'] = coin_p1
            data['P2'] = coin_p2
            data['cutoff'] = cutoff
            data['flips'] = icl_flips

            # Calculate expected distributions
            heads_prob = bias_percentage / 100.0
            tails_prob = 1.0 - heads_prob
            data['needle_expected_dists'].append([heads_prob, tails_prob])

            # Now generate the adjusted input prompts
            adjusted_input_prompts = []

            for prompt in input_prompts:
                
                # Build the adjusted input prompt using the specified format
                icl_examples_text = ', then '.join(icl_flips)
                
                if len(icl_flips) > 0:
                    adjusted_prompt = f"{prompt}{icl_examples_text}, then "
                else:
                    adjusted_prompt = f"{prompt}"
                    
                adjusted_input_prompts.append(adjusted_prompt)

            # Assign the adjusted input prompts to the data
            data['input_prompt'] = adjusted_input_prompts

            # Write the data to a YAML file
            filename = f"bias_{bias_percentage}.yaml"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                yaml.dump(data, f)
                
        # Generate the coin flips for the next iteration.
        # P1 first. 
        if icl_length < cutoff:
            flip = random.choices(choices, weights=[coin_p1, 1-coin_p1])[0]
            
        # Then P2 after cutoff.
        else:
            flip = random.choices(choices, weights=[coin_p2, 1-coin_p2])[0]
        
        icl_flips.append(flip)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create posterior chain coin flip configs",
    )

    parser.add_argument('--theta-start', type=int, help="The starting range for the thetas to sweep over", required=True, default=0)
    parser.add_argument('--theta-end', type=int, help="The end range for the thetas to sweep over", required=True, default=110)
    parser.add_argument('--theta-step', type=int, help="The step size for the list of thetas to sweep over", required=True, default=10)
    parser.add_argument('--theta-one', type=float, help="Value for theta_1 (ignored if cutoff == 0)", required=True, default=0.25)
    parser.add_argument('--theta-two', type=float, help="Value for theta_2 (ignored if cutoff >= max(icl_examples_list))", required=True, default=0.75)
    parser.add_argument('--cutoff', type=int, help="Where in the sequence to switch from theta_1 to theta_2", required=True, default=500)
    parser.add_argument('--choices', nargs='+', help="The available choices (the code supports two right now)", default=["heads", "tails"])
    parser.add_argument('--icl-start', type=int, help="The starting range for the ICL length to sweep over", required=True, default=0)
    parser.add_argument('--icl-end', type=int, help="The starting range for the ICL length to sweep over", required=True, default=1001)
    parser.add_argument('--experiment-base-dir', type=str, help="Where to save the config files", required=True)
    parser.add_argument('--seed', type=int, help="Seed for reproducibility", default=42)

    args = parser.parse_args()
    generate_configs(args)