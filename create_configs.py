import yaml
import argparse
import os

# Common settings for all experiments.
CONTEXT_LENGTHS = [length for length in range(10, 110, 10)] + [200, 400, 800, 1000]
DOCUMENT_DEPTHS = [d / 10 for d in range(0, 11, 1)]

def create_coin_flip_config(args):

    # Global settings.
    cfg = {
        'experiment_dir': os.path.join(args.exp_dir, 'coin_flip'),
        'input_prompt': "I tossed a coin and it landed on ",
        'instruct_prompt': "Please toss a coin and tell me whether it landed on head or tails.",
        'choices': ['heads', 'tails'],
        'context_lengths': CONTEXT_LENGTHS,
        'document_depths': DOCUMENT_DEPTHS,
        'default_expected_dist': [0.5, 0.5]
    }

    # Create bias needles with expected distributions.
    bias_needles = []
    instruct_bias_needles = []
    expected_dists = []

    for bias in range(0, 110, 10):
        bias_needles.append(f"When I flip coins they land on heads {bias}% of the time.")
        instruct_bias_needles.append(f"When you flip coins, they land on heads {bias}% of the time.")
        expected_dists.append([round(bias/100, 1), round(1 - bias/100, 1)])

    # Add to config.
    cfg['bias_needles'] = bias_needles
    cfg['instruct_bias_needles'] = instruct_bias_needles
    cfg['needle_expected_dists'] = expected_dists

    # Save config.
    with open(os.path.join(args.config_dir, 'coin_flip.yaml'), 'w') as f:
        yaml.dump(cfg, f)

def create_die_config(args):

    # Global settings.
    cfg = {
        'experiment_dir': os.path.join(args.exp_dir, 'die_roll'),
        'input_prompt': "I rolled a die and it landed on ",
        'instruct_prompt': "Please roll a six-sided die and tell me what number it landed on.",
        'choices': ['1', '2', '3', '4', '5', '6'],
        'context_lengths': CONTEXT_LENGTHS,
        'document_depths': DOCUMENT_DEPTHS,
        'default_expected_dist': [1, 1, 1, 1, 1, 1]
    }

    # Create bias needles with expected distributions.
    bias_needles = []
    instruct_bias_needles = []
    expected_dists = []

    for num in range(1, 7):
        for bias in [60, 80]:

            # Bias needle.
            bias_needles.append(f"When I roll dice they land on {num} {bias}% of the time.")
            instruct_bias_needles.append(f"Whenever you roll dice they land on {num} {bias}% of the time.")

            # Create expected distribution.
            expected_dist = [0] * 6
            expected_dist[num-1] = bias / 100

            for i in range(1, 7):
                if i != num:
                    expected_dist[i-1] = (100 - bias) / 500

            expected_dists.append(expected_dist)

    # Add to config.
    cfg['bias_needles'] = bias_needles
    cfg['instruct_bias_needles'] = instruct_bias_needles
    cfg['needle_expected_dists'] = expected_dists

    # Save config.
    with open(os.path.join(args.config_dir, 'die_roll.yaml'), 'w') as f:
        yaml.dump(cfg, f)


def create_random_number_config(args):

    # Global settings.
    cfg = {
        'experiment_dir': os.path.join(args.exp_dir, 'random_number'),
        'input_prompt': "I picked a random number between 1 and 10 and it was ",
        'instruct_prompt': "Please pick a random number between 1 and 10.",
        'choices': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'context_lengths': CONTEXT_LENGTHS,
        'document_depths': DOCUMENT_DEPTHS,
        'default_expected_dist': [0.1] * 10
    }

    # Create bias needles with expected distributions.
    bias_needles = []
    instruct_bias_needles = []
    expected_dists = []

    # Bias needles.
    biases = ['even', 'odd']

    for bias in biases:

        # Bias needles.
        bias_needles.append(f'When I pick random numbers, they are always {bias}.')
        instruct_bias_needles.append(f'When you pick random numbers, they are always {bias}.')

        # Determine if even or odd.
        even_odd = 0 if bias == 'even' else 1

        expected_dist = [0] * 10

        for i in range(1, 11):
            if i % 2 == even_odd:
                expected_dist[i-1] = 0.2
            else:
                expected_dist[i-1] = 0.0

        expected_dists.append(expected_dist)

    # Add to config.
    cfg['bias_needles'] = bias_needles
    cfg['instruct_bias_needles'] = instruct_bias_needles
    cfg['needle_expected_dists'] = expected_dists

    # Save config.
    with open(os.path.join(args.config_dir, 'random_number.yaml'), 'w') as f:
        yaml.dump(cfg, f)

if __name__ == '__main__':

    # Arguments.
    parser = argparse.ArgumentParser(description='Create configuration files.')
    parser.add_argument('config_dir', type=str, help='Output directory to place config files in.')
    parser.add_argument('exp_dir', type=str, help='Parent directory to specify for experiments in config file.')
    args = parser.parse_args()

    # Create the configuration files.
    create_coin_flip_config(args)
    create_die_config(args)
    create_random_number_config(args)
