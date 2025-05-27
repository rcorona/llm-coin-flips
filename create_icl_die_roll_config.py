import yaml
import random
import os

# Define the parameters
bias_percentages = [0, 20, 40, 60, 80, 100]  # 0%, 20%, ..., 100%

choices = ["1", "2", "3", "4", "5", "6"]

bias_targets = ["1", "2", "3", "4", "5", "6"]

context_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1000]

default_expected_dist = [1/6]*6

document_depths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# List of icl_examples values
icl_examples_list = [1, 3, 5, 10, 20, 100]

# Base experiment directory
base_experiment_dir = 'exps/contextual/die_roll_icl_v2'

instruct_prompt = [
    "Please complete this sentence: I just rolled a six-sided die, and it showed ",
    "Finish this sentence: The result of my six-sided die roll is ",
    "Complete the sentence: After rolling the six-sided die, it landed on ",
    "Fill in the rest: I rolled a six-sided die, and the outcome was ",
    "Complete the phrase: The six-sided die roll resulted in "
]

input_prompts = [
    "I just rolled a six-sided die, and it showed ",
    "The result of my six-sided die roll is ",
    "After rolling the six-sided die, it landed on ",
    "I rolled a six-sided die, and the outcome was ",
    "The six-sided die roll resulted in ",
    "After tossing the six-sided die, it ended up on ",
    "I tossed a six-sided die, and it fell on ",
    "The six-sided die I rolled landed on ",
    "The six-sided die I tossed resulted in ",
    "Following the six-sided die roll, it showed ",
    "The six-sided die fell on ",
    "The roll of the six-sided die resulted in ",
    "I rolled the six-sided die, and it settled on ",
    "The result after rolling the six-sided die is ",
    "The outcome of my six-sided die roll is ",
    "I tossed the six-sided die, and the outcome is ",
    "The result of my six-sided die toss is ",
    "I rolled the six-sided die, and it came up ",
    "The six-sided die came to rest on ",
    "After rolling, the six-sided die showed ",
    "The toss of the six-sided die revealed ",
    "I rolled the six-sided die, and it turned up ",
    "The six-sided die toss ended with ",
    "After tossing the six-sided die, it showed ",
    "The six-sided die rolled over to ",
    "After rolling, the six-sided die settled on ",
    "My six-sided die toss resulted in ",
    "The outcome of my six-sided die roll turned out to be ",
    "I rolled the six-sided die, and its final position was ",
    "The six-sided die fell, showing ",
    "I tossed the six-sided die, and it landed showing ",
    "Following the toss, the six-sided die showed ",
    "The roll resulted in the six-sided die landing on ",
    "The six-sided die toss revealed ",
    "The outcome of the six-sided die landing is ",
    "After tossing, the six-sided die landed on ",
    "I rolled the six-sided die and saw it land on ",
    "After the roll, the six-sided die showed ",
    "The result of tossing the six-sided die was ",
    "When I rolled the six-sided die, it landed on ",
    "The six-sided die showed this side after the roll: ",
    "The roll of the six-sided die ended with ",
    "After tossing, the six-sided die fell to show ",
    "The result of my toss came out as ",
    "The toss of the six-sided die came to rest on ",
    "The six-sided die after the roll landed on ",
    "I rolled the six-sided die, and it ended on ",
    "The result of the six-sided die toss ended up being ",
    "I rolled a six-sided die, and its final side was ",
    "The six-sided die roll showed the result: "
]

# For reproducibility
random.seed(42)

# Now we will loop over the icl_examples list
for icl_examples in icl_examples_list:
    # Create output directory for this icl_examples value
    output_dir = f'configs/exps/contextual/die_roll_icl_v2/icl_{icl_examples}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Now we will generate separate YAML files for each bias target and bias percentage
    for bias_target in bias_targets:
        for bias_percentage in bias_percentages:
            # Create data dictionary
            data = {}
            data['bias_needles'] = []
            data['choices'] = choices
            data['context_lengths'] = context_lengths
            data['default_expected_dist'] = default_expected_dist
            data['document_depths'] = document_depths

            # Adjust the experiment_dir to include icl_examples, bias_target, and bias_percentage
            data['experiment_dir'] = os.path.join(base_experiment_dir, f'icl_{icl_examples}', f'bias_{bias_target}_{bias_percentage}')

            data['instruct_prompt'] = instruct_prompt
            data['input_prompt'] = []
            data['instruct_bias_needles'] = []
            data['needle_expected_dists'] = []

            # Create bias needle and instruct bias needle
            bias_needle = f"When I roll dice, they land on {bias_target} {bias_percentage}% of the time."
            instruct_bias_needle = f"When you roll dice, they land on {bias_target} {bias_percentage}% of the time."
            data['bias_needles'].append(bias_needle)
            data['instruct_bias_needles'].append(instruct_bias_needle)

            # Calculate expected distributions
            p = bias_percentage / 100.0
            other_prob = (1.0 - p) / (len(choices) - 1)
            probs = []
            for choice in choices:
                if choice == bias_target:
                    probs.append(p)
                else:
                    probs.append(other_prob)
            data['needle_expected_dists'].append(probs)

            # Now generate the adjusted input prompts
            adjusted_input_prompts = []

            for prompt in input_prompts:
                # Generate icl_examples number of die rolls according to the biased probability
                icl_rolls = []
                for _ in range(icl_examples):
                    roll = random.choices(choices, weights=probs)[0]
                    icl_rolls.append(roll)
                # Build the adjusted input prompt
                icl_examples_text = ', then '.join(icl_rolls)
                adjusted_prompt = f"{prompt}{icl_examples_text}, then "
                adjusted_input_prompts.append(adjusted_prompt)

            # Assign the adjusted input prompts to the data
            data['input_prompt'] = adjusted_input_prompts

            # Write the data to a YAML file
            filename = f"bias_{bias_target}_{bias_percentage}.yaml"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                yaml.dump(data, f)
