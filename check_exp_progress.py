import argparse
import pickle
import os
import yaml

# List of instruct models we'll use.
from token_prob_experiments_v2 import INSTRUCT_MODELS

def num_pivot_table_experiments(args, use_needle):

    # Determine whether to use bias needles or not.
    if use_needle:
        bias_needles = args.bias_needles
        doc_depths = args.document_depths
    else:
        bias_needles = [""]
        doc_depths = [0]

    # Iterate over context lengths, document depths, and bias needles.
    return len(args.context_lengths) * len(doc_depths) * len(bias_needles)

def get_exp_progress(args, model_name):
    """
    Report the percentage completion of a given experiment for a given config.
    """

    complete = {
        'unconditional_point': 0.0,
        'biased_point': 0.0,
        'pivot_table': 0.0,
        'needle_pivot_table': 0.0
    }

    # Build model dir.
    model_dir = os.path.join(args.experiment_dir, model_name)

    # Load collected stats if they exist.
    stats_path = os.path.join(model_dir, 'stats.pkl')

    # Check whether model is in list of instruct models.
    instruct_model = model_name in INSTRUCT_MODELS

    # Determine which bias needles to use.
    if instruct_model:
        args.bias_needles = args.instruct_bias_needles

    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
        except Exception as ex:
            print(f"Error loading stats for {model_name}: {ex}")
            return complete

        # Count number of completed experiments per type.
        if 'unconditional_point' in stats:
            complete['unconditional_point'] = 1.0

        if 'biased_point' in stats:
            complete['biased_point'] = len(stats['biased_point']) / len(args.bias_needles)

        if 'pivot_table' in stats:
            complete['pivot_table'] = len(stats['pivot_table']) / num_pivot_table_experiments(args, use_needle=False)

        if 'needle_pivot_table' in stats:
            complete['needle_pivot_table'] = len(stats['needle_pivot_table']) / num_pivot_table_experiments(args, use_needle=True)

    return complete

if __name__ == '__main__':

    # Arguments.
    parser = argparse.ArgumentParser('Random Needle Experiment Verification')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Path to experiment directory.')
    args = parser.parse_args()

    # Load config file.
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

        # Update args with config.
        for key, value in config.items():
            setattr(args, key, value)

    # Path to where we'll store stats.
    for model_name in os.listdir(args.experiment_dir):
        complete = get_exp_progress(args, model_name)

        # Print completion stats.
        print(f"Model: {model_name}")
        print(f"Unconditional Point: {100.0 * complete['unconditional_point']}%")
        print(f"Biased Point: {100.0 * complete['biased_point']}%")
        print(f"Pivot Table: {100.0 * complete['pivot_table']}%")
        print(f"Needle Pivot Table: {100.0 * complete['needle_pivot_table']}%")
        print('\n\n')
