import yaml
import random
import os

# Define the parameters
bias_percentages = list(range(0, 110, 10))  # 0%, 10%, ..., 100%

choices = ["heads", "tails"]

context_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1000]

default_expected_dist = [0.5, 0.5]

document_depths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# List of icl_examples values
# icl_examples_list = [1, 3, 5, 10, 20, 100]
icl_examples_list = [500]                               # only using 100 for now
icl_cutoff_list = [50, 100, 150, 200, 250, 300, 350, 400, 450]       # for d1 CUTOFF d2

# Base experiment directory
base_experiment_dir = 'exps/contextual/coin_flip_icl_double_dist'

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
random.seed(42)

# Now we will loop over the icl_examples list
for icl_examples in icl_examples_list:
    for cutoff in icl_cutoff_list:

        # Create output directory for this icl_examples value
        output_dir = f'configs/exps/contextual/coin_flip_icl_double_dist/icl_{icl_examples}/cutoff_{cutoff}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Now we will generate separate YAML files for each bias percentage
        for bias_percentage_1 in bias_percentages:
            for bias_percentage_2 in bias_percentages:

                # if bias_percentage_1 != bias_percentage_2:
                #     continue

                # Create data dictionary for this bias percentage and icl_examples
                data = {}
                data['bias_needles'] = []
                data['choices'] = choices
                data['context_lengths'] = context_lengths
                data['default_expected_dist'] = default_expected_dist
                data['document_depths'] = document_depths

                # Adjust the experiment_dir to include icl_examples
                data['experiment_dir'] = os.path.join(base_experiment_dir, f'icl_{icl_examples}', f'cutoff_{cutoff}', f'{bias_percentage_1}_{bias_percentage_2}')

                data['instruct_prompt'] = instruct_prompt
                data['input_prompt'] = []
                data['instruct_bias_needles'] = []
                data['needle_fake_dist'] = []           # fake dist is the first distribution that only occurs in ICL until CUTOFF
                data['needle_expected_dists'] = []      # expected dist is the second distribution that we want to see at the end

                # Create bias needle and instruct bias needle
                bias_needle = f"When I flip coins they land on heads {bias_percentage_2}% of the time."
                instruct_bias_needle = f"When you flip coins, they land on heads {bias_percentage_2}% of the time."
                data['bias_needles'].append(bias_needle)
                data['instruct_bias_needles'].append(instruct_bias_needle)

                # Calculate expected distributions
                heads_prob_1 = bias_percentage_1 / 100.0
                tails_prob_1 = 1.0 - heads_prob_1
                data['needle_fake_dist'].append([heads_prob_1, tails_prob_1])

                heads_prob_2 = bias_percentage_2 / 100.0
                tails_prob_2 = 1.0 - heads_prob_2
                data['needle_expected_dists'].append([heads_prob_2, tails_prob_2])

                # Now generate the adjusted input prompts
                adjusted_input_prompts = []

                probs1 = [heads_prob_1, tails_prob_1]
                probs2 = [heads_prob_2, tails_prob_2]
                for prompt in input_prompts:
                    # Generate icl_examples number of coin flips according to the bias probability
                    icl_flips = []
                    for i in range(icl_examples):
                        probs = probs1 if i < cutoff else probs2
                        flip = random.choices(choices, weights=probs)[0]
                        icl_flips.append(flip)

                    # Build the adjusted input prompt using the specified format
                    icl_examples_text = ', then '.join(icl_flips)
                    adjusted_prompt = f"{prompt}{icl_examples_text}, then "
                    adjusted_input_prompts.append(adjusted_prompt)

                # Assign the adjusted input prompts to the data
                data['input_prompt'] = adjusted_input_prompts

                # Write the data to a YAML file
                filename = f"bias_{bias_percentage_1}_{bias_percentage_2}.yaml"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    yaml.dump(data, f)
