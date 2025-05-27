from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets
import torch        
from torch.utils.data import DataLoader
import torch.nn.functional as F
import spacy

import argparse
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
import pickle
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def create_filler_dataset(args, tokenizer, context_length):
    """
    Preprocesses filter dataset into sentences and then uses them to populate
    a dataset with the desired number of examples for statistical analysis. 
    """
    
    # Will contain lists of tokenized sentences for each sample. 
    samples = []
    curr_len = 0
    curr_sample = []
    
    # Load sentencizer from Spacy. 
    nlp = spacy.load("en_core_web_sm")

    # Load dataset.
    dataset = datasets.load_dataset(path=args.filler_dataset, data_dir=args.dataset_subset)[args.dataset_split]
    
    # Iterate over dataset and tokenize sentences.
    print('Tokenizing filler dataset...')
    pbar = tqdm(dataset, total=args.n_samples)
    
    for example in pbar:
        
        # Split example into sentences.
        doc = nlp(example['text'])

        # Tokenize each sentence.
        for sent in doc.sents: 
                
            # Tokenize sentence.
            tokens = tokenizer(sent.text)['input_ids']
            
            # Add to current sample.
            curr_sample.append(tokens)
            curr_len += len(tokens)
            
            # Switch to new sample if we've reached the desired length.
            if curr_len >= context_length:
                samples.append(curr_sample)
                curr_sample = []
                curr_len = 0
                pbar.update(1)
                
                # Return if we've reached the desired number of samples.
                if len(samples) == args.n_samples:
                    return samples
                
    # We should never reach this point. 
    raise ValueError('Not enough samples in filler dataset to reach desired number of samples!')

def inject_needle(dataset, needle, document_depth):
    """
    Injects a biasing needle into the filler context at a specified depth. 
    """
        
    # Inject needle into each sample.
    for sample in dataset:
        
        # Get total length of document. 
        doc_len = sum([len(sent) for sent in sample])
        
        # Compute number of tokens that would get us to desired depth.
        token_depth = int(doc_len * document_depth)
        
        # Determine which document to inject needle after. 
        inject_idx = 0
        len_so_far = 0
        
        # Find the sentence to inject needle after.
        for sent in sample:
            
            # Update length so far.
            len_so_far += len(sent)
            inject_idx += 1
            
            # If we've reached the desired depth, inject needle after this sentence.
            if len_so_far >= token_depth:
                sample.insert(inject_idx, needle)
                break
        
    return dataset

def create_random_needle_dataset(args, tokenizer, context_length, document_depth):
    """
    Creates random needle dataset consisting of a single prompt and multiple answer choices. 
    Optionally will add a "biasing needle" to the prompt along with filler context to bias the choices. 
    The needle will be injected somewhere in the filler context based on the document depth parameter.
    """
    # cot_prompt = "Let's think step by step, "
    cot_prompt = ""
    cot_needle = tokenizer(cot_prompt)['input_ids']
    # Format needle string as needed. 
    if len(args.bias_needle) > 0:
        bias_needle = tokenizer(args.bias_needle)['input_ids']
    else: 
        bias_needle = []
    
    # Special case of no context, only a single prompt can be created. 
    if context_length == 0:
        
        # Create dataset with only the input prompt and bias needle if provided. 
        dataset = [{'choices': [bias_needle + tokenizer(args.input_prompt + choice)['input_ids'] for choice in args.choices]}]
    else:
        
        # First create filler dataset with n_samples each with context_length tokens.
        filler = create_filler_dataset(args, tokenizer, context_length)
    
        # Inject bias needle into each filler context sample. 
        filler = inject_needle(filler, bias_needle, document_depth)
        
        # Tokenize input prompt and choices. 
        input_prompt = tokenizer(args.input_prompt)['input_ids']
        choices = [tokenizer(choice)['input_ids'] for choice in args.choices]
        
        # Create dataset with input prompt and choices, keep track of lengths of filler contexts so we can cache computation over choices. 
        dataset = []
        
        # Final dataset will contain all possible combinations of filler context and choices.
        print('Creating dataset with all samples and choice combinations...')
        for sample in tqdm(filler): 

            # Add input prompt. 
            sample.append(input_prompt)
            sample.append(cot_needle)
            
            # Add a datapoint containing all combinations of sample and choice. 
            datapoint = []
            for choice in choices:
                datapoint.append([i for i in itertools.chain.from_iterable(sample + [choice])])

            dataset.append({'choices': datapoint})
    print(dataset)
    return dataset

def run_random_needle_exp(args, model, tokenizer, context_len, doc_depth):
    
    # Create dataset. 
    dataset = create_random_needle_dataset(args, tokenizer, context_len, doc_depth)
        
    # Create dataloader. 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=0)
        
    # Collect sequence probabilities.
    n_choices = len(args.choices)
    all_seq_probs = np.zeros((len(dataset), n_choices))
    kl_divergences = np.zeros((len(dataset), n_choices))
        
    # Compute probabilities. 
    for b_idx, batch in enumerate(tqdm(dataloader)):

        # Put on gpu. 
        inputs = {k: v.cuda() for k, v in batch.items()}

        # Forward pass. 
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Compute token log probabilities. 
        all_log_probs = F.log_softmax(outputs.logits, dim=-1)
        
        # Get log probs of input tokens in each sequence. 
        n_log_probs = batch['input_ids'].shape[0] * batch['input_ids'].shape[1]
        token_log_probs = all_log_probs.view(n_log_probs, -1)[torch.arange(n_log_probs), batch['input_ids'].flatten()].view(batch['input_ids'].shape)
 
        # Mask out padding. 
        token_log_probs = token_log_probs * inputs['attention_mask']
        
        # Sum log probabilities to get sequence log probabilities.
        seq_log_probs = token_log_probs.sum(dim=-1)
        
        # Normalize by sequence length.
        seq_log_probs = seq_log_probs / inputs['attention_mask'].sum(dim=-1)
        
        # Uncollapse so we get log probabilities for each answer choice sequence per dataset example.
        seq_log_probs = seq_log_probs.view(-1, n_choices)
        
        # Normalize to get a distribution. 
        norm_seq_probs = torch.softmax(seq_log_probs, dim=-1)
        
        # Compute point-wise KL divergence.
        target_dist = expected_dist.view(1, -1).repeat(norm_seq_probs.size(0), 1)
        kl_div = F.kl_div(torch.log(norm_seq_probs), target_dist, reduction='none', log_target=True)
        
        # Collect results.
        start = b_idx * args.batch_size
        end = start + norm_seq_probs.size(0)
        all_seq_probs[start:end] = norm_seq_probs.cpu().numpy()
        kl_divergences[start:end] = kl_div.cpu().numpy()

    # Marginalize out over samples. 
    choice_probs = np.mean(all_seq_probs, axis=0)
    
    # Compute KL divergence from expected distribution. 
    kl_div = np.sum(kl_divergences) / kl_divergences.shape[0]
    
    # Consolidate stats and return them. 
    stats = {
        'choice_probs': choice_probs,
        'kl_div': kl_div,
    }
    
    return stats

    
def gen_point_outputs(args, stats):
    """
    Generate experiment output for a single data point (i.e. no context or document depth used.)
    """
    
    # Unpack stats. 
    point_stats = stats[(0, None)]
    choice_probs = point_stats['choice_probs']
    kl_div = point_stats['kl_div']
    
    # Bar graph probabilities for each choice.
    plt.bar(range(len(args.choices)), choice_probs * 100)
    plt.title(f'Conflated Probabilities for {args.model}, KL divergence: {kl_div:.4f}')
    plt.xticks(range(len(args.choices)), args.choices)
    plt.xlabel('Answer choice')
    plt.ylabel('Probability')
    plt.text(0, 0.5, f'Input prompt: "{args.input_prompt}"', fontsize=8)
    
    # Save figure. 
    save_path = os.path.join(args.experiment_dir, f'{args.config.split("/")[-1].split(".")[0]}')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path,f'{args.model.replace("/", "_")}_bargraph.png')
    plt.savefig(save_path)

def gen_multipoint_outputs(args, stats):
    """
    Generate experiment output for multiple data points (i.e. context and document depth used.)
    
    Pivot table code based on: https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/viz/CreateVizFromLLMTesting.ipynb
    """
    
    # Format data as pandas dataframe.
    data = []
    
    for (context_len, doc_depth), point_stats in stats.items():
        data.append({
            'Context Length': context_len,
            'Depth': doc_depth,
            'KL Divergence': point_stats['kl_div']
        }) 

    df = pd.DataFrame(data)

    # Create pivot table to visualize data. 
    pivot_table = pd.pivot_table(df, values='KL Divergence', index=['Context Length', 'Depth'], aggfunc=np.mean).reset_index()
    pivot_table = pivot_table.pivot(index="Depth", columns="Context Length", values="KL Divergence")

    # Color map. 
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'KL Divergence'},
    )

    # More aesthetics
    plt.title(f'Pivot table for {args.model}')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Save figure in experiment directory.
    save_path = os.path.join(args.experiment_dir, f'{args.config.split("/")[-1].split(".")[0]}')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path,f'{args.model.replace("/", "_")}_bargraph.png')
    plt.savefig(save_path)

if __name__ == '__main__':
    
    # Arguments. 
    parser = argparse.ArgumentParser('Random Needle Experiments')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('model_config', type=str, help='Path to model config file.')

    # For command line use, but can be overwritten by config file.
    parser.add_argument('--model', type=str, default="google/gemma-2-2b", help="Model to use.")
    parser.add_argument('--input_prompt', type=str, default="I tossed a coin, and it landed on ", help="Prompt to use.")
    parser.add_argument('--choices', type=str, nargs='+', default=['heads', 'tails'], help="Answer choices.")
    parser.add_argument('--bias_needle', type=str, default=None, help="Bias prompt to inject as needle for experiment.")
    parser.add_argument('--expected_dist', type=float, nargs='+', default=[0.5, 0.5], help="Expected distribution.")
    parser.add_argument('--filler_dataset', type=str, default="Salesforce/wikitext", help="Dataset to use for filler context.")
    parser.add_argument('--dataset_subset', type=str, default="wikitext-103-v1", help="Subset of dataset to use.")
    parser.add_argument('--dataset_split', type=str, default="train", help="Split of dataset to use.")
    parser.add_argument('--context_length', type=int, default=0, help="Length of context to use.")
    parser.add_argument('--n_samples', type=int, default=1, help="Number of samples to use to compute statistics.")
    parser.add_argument('--document_depth', type=float, default=0.5, help="Percentage in [0, 1] of depth in document to inject needle.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size.")
    parser.add_argument('--IG_prior_dist', type=float, nargs='+', default=[0.5, 0.5], help="Prior distribution for information gain measurements.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    
    # Random seed. 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # TODO Make sure that we're concatenating choices to input prompt at string level and not at token level!
    
    # Load config files. 
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
        # Update args with config.
        for key, value in config.items():
            setattr(args, key, value)

    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # TODO generalize to iteration over models. 
    args.model = model_config['models'][0]
    
    # Create experiment directory if it doesn't exist.
    os.makedirs(os.path.join(args.experiment_dir), exist_ok=True)
    
    # Load model.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        attn_implementation="eager", # TODO add inference-time modification to allow flash-attention. 
        torch_dtype=torch.float32,
        quantization_config=bnb_config,
    )
    
    # Probability distribution we're expecting, normalize in case it doesn't sum to 1 already.
    expected_dist = np.asarray(args.expected_dist) / np.sum(args.expected_dist) 
    expected_dist = torch.log(torch.Tensor(expected_dist).float().cuda())
    
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Add padding token if needed. 
    if tokenizer.pad_token_id is None: 
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.padding_side = 'left'
    
    # Collate function for dataloader.
    def collate_fn(batch):
        
        # Unpack batch. 
        samples = [item['choices'] for item in batch]

        # Collapse examples into batch of single strings. 
        samples = {'input_ids': list(itertools.chain.from_iterable(samples))}
        
        # Tokenize.
        inputs = tokenizer.pad(samples, return_tensors='pt', padding=True)
                
        return inputs
    
    # Iterate over context lengths and document depths if provided. 
    if args.use_context: 
        context_lengths = args.context_lengths
        document_depths = args.document_depths
    
    # Otherwise will not use. 
    else: 
        context_lengths = [0]
        document_depths = [None]
    
    # Path to where we'll store stats. 
    stats_path = os.path.join(args.experiment_dir, 'stats.pkl')
    
    # Pre-load stats if we already have some results to avoid recomputing.
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
    else: 
        stats = {}
    # Collect statistics for each context length and document depth.
    for context_len in context_lengths: 
        for doc_depth in document_depths: 
            
            # Skip if we've already computed this.
            if (context_len, doc_depth) in stats:
                continue
            
            # Otherwise, run experiment and store results.
            stats[(context_len, doc_depth)] = run_random_needle_exp(args, model, tokenizer, context_len, doc_depth)
            
            # Save in case we get interrupted. 
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
            
    # Generate experiment output/plots based on whether we're using context or not.
    if args.use_context: 
        gen_multipoint_outputs(args, stats)
    
    else: 
        # Generate output for single data point case. 
        gen_point_outputs(args, stats)