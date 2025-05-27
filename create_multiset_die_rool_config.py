import yaml

# Define the range of biases (from 0% to 100% in steps of 20%)
bias_percentages = [0, 20, 40, 60, 80, 100]

# Define the sides of the die
dice_sides = ['1', '2', '3', '4', '5', '6']

# Prepare the YAML configuration dictionary
yaml_config = {
    'bias_needles': [],
    'choices': dice_sides,
    'context_lengths': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1000],
    'default_expected_dist': [1] * 6,
    'document_depths': [i / 10.0 for i in range(11)],
    'experiment_dir': 'exps/contextual/die_roll_multiset',
    'input_prompt': [
        "I rolled a six-sided die and it landed on ",
        "I tossed a six-sided die, and it came up with ",
        "The result of my six-sided die roll is ",
        "After rolling the six-sided die, it showed ",
        "The six-sided die landed on ",
        "The outcome of the six-sided die roll is ",
        "After tossing the six-sided die, it landed on ",
        "I rolled a six-sided die and the result was ",
        "The six-sided die roll revealed ",
        "The six-sided die landed showing ",
        "I threw the six-sided die, and it came up with ",
        "The six-sided die settled on ",
        "The six-sided die stopped at ",
        "My six-sided die roll resulted in ",
        "The final number of the six-sided die roll was ",
        "The six-sided die roll ended with it showing ",
        "After tossing, the six-sided die came to rest on ",
        "I tossed the six-sided die, and it showed ",
        "The six-sided die roll resulted in it landing on ",
        "The six-sided die turned up with the number ",
        "Following the roll, the six-sided die showed ",
        "I rolled the six-sided die, and it landed on ",
        "The number on the six-sided die was ",
        "I threw a six-sided die and it came to rest on ",
        "The result of rolling the six-sided die is ",
        "The final outcome of the six-sided die roll is ",
        "The six-sided die showed the number ",
        "The roll of the six-sided die produced the number ",
        "I rolled the six-sided die and it ended up showing ",
        "After tossing the six-sided die, it revealed ",
        "I threw the six-sided die, and the result was ",
        "After rolling, the six-sided die showed ",
        "The toss resulted in the six-sided die showing ",
        "The six-sided die flipped over and landed on ",
        "The final number after rolling the six-sided die was ",
        "After rolling, the six-sided die stopped on ",
        "I rolled the six-sided die, and the number was ",
        "After tossing, the six-sided die settled on ",
        "The six-sided die revealed the number ",
        "The final result of the six-sided die roll was ",
        "I tossed the six-sided die and it landed on ",
        "The six-sided die rolled to show ",
        "The outcome after rolling the six-sided die is ",
        "After tossing the six-sided die, it stopped on ",
        "I rolled the six-sided die, and it displayed ",
        "The final outcome of rolling the six-sided die is ",
        "The number the six-sided die landed on was ",
        "I tossed the six-sided die, and it settled on ",
        "After rolling, the six-sided die revealed the number ",
        "The result of the six-sided die roll turned out to be "
    ],
    'instruct_bias_needles': [],
    'instruct_prompt':  [
        "Please roll a six-sided die and tell me which number it landed on.",
        "Roll a six-sided die and let me know what number it shows.",
        "Please roll a six-sided die and share the result.",
        "Toss a six-sided die and tell me the number it came up with.",
        "Roll a six-sided die and give me the number it shows.",
        "Please roll the six-sided die and report the number that it lands on.",
        "Throw a six-sided die and tell me which number it shows."
    ],
    'needle_expected_dists': []
}

# Generate bias_needles and instruct_bias_needles
for side in dice_sides:
    for bias in bias_percentages:
        bias_statement = f"When I roll dice they land on {side} {bias}% of the time."
        yaml_config['bias_needles'].append(bias_statement)

        instruct_bias_statement = f"Whenever you roll dice they land on {side} {bias}% of the time."
        yaml_config['instruct_bias_needles'].append(instruct_bias_statement)

        # Calculate the expected distribution
        bias_prob = bias / 100.0
        other_prob = (1.0 - bias_prob) / (len(dice_sides) - 1)
        expected_dist = []
        for s in dice_sides:
            if s == side:
                expected_dist.append(bias_prob)
            else:
                expected_dist.append(other_prob)
        yaml_config['needle_expected_dists'].append(expected_dist)

# Save the configuration to a YAML file
with open('/home/davidchan/Repos/random_needles/configs/exps/contextual/die_roll_multiset_v2.yaml', 'w') as file:
    yaml.dump(yaml_config, file, default_flow_style=False)

print("YAML configuration file '/home/davidchan/Repos/random_needles/configs/exps/contextual/die_roll_multiset_v2.yaml' has been generated.")
