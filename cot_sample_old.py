from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import PaddingStrategy
import datasets
import torch        
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import spacy
from scipy.special import logsumexp
import re
import argparse
import pdb
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import itertools
import yaml
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Collate function for dataloader.

# Local imports. 
from gen_figures import gen_point_outputs, gen_multipoint_outputs

# List of instruct models. 
INSTRUCT_MODELS = [
    'google/gemma-2-2b-it', 
    'meta-llama/Llama-3.1-8B-Instruct', 
    'meta-llama/Llama-3.1-70B-Instruct',
    'meta-llama/Llama-3.1-405B-Instruct',
    'microsoft/Phi-3.5-MoE-instruct', 
    'mistralai/Mistral-7B-Instruct-v0.3', 
    'allenai/OLMoE-1B-7B-0924-Instruct'
]

def collate_fn(batch):
    
    # Unpack batch. 
    samples = [item['choices'] for item in batch]

    # Collapse examples into batch of single strings. 
    samples = {'input_ids': list(itertools.chain.from_iterable(samples))}
    
    # Tokenize.
    inputs = tokenizer.pad(samples, return_tensors='pt', padding=True)
            
    return inputs
    
def preprocess_expected_dist(expected_dist):
    """
    Convert a list expected distribution into a numpy array in logspace. 
    """
    expected_dist = np.asarray(expected_dist).astype(np.float32)
                
    # Normalize expected dist and convert to tensor in logspace. 
    expected_dist = expected_dist / expected_dist.sum()
    expected_dist = torch.log(torch.Tensor(expected_dist).float().cuda())
    
    return expected_dist
    
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

def create_cot_dataset(dataset, tokenizer):
    samples = []
    for seq in dataset:
        samples.append([tokenizer(seq)['input_ids']])
    return samples
    
def create_dataset(args, model, tokenizer, context_length, document_depth, bias_needle):
    """
    Creates random needle dataset consisting of a single prompt and multiple answer choices. 
    Optionally will add a "biasing needle" to the prompt along with filler context to bias the choices. 
    The needle will be injected somewhere in the filler context based on the document depth parameter.
    """
    # cot_prompt = "Let's think step by step, "
    # Format needle string as needed. 
    datasets = random_needle_cot_dataset(args, model, tokenizer, context_length, document_depth, bias_needle)
    if len(bias_needle) > 0:
        bias_needle = tokenizer(bias_needle)['input_ids']
    else: 
        bias_needle = []
    
    # Special case of no context, only a single prompt can be created. 
    if context_length == 0:
        
        # Create dataset with only the input prompt and bias needle if provided. 
        dataset = [{'choices': [bias_needle + tokenizer(args.input_prompt + choice)['input_ids'] for choice in args.choices]}]
    else:
        
    
        # Inject bias needle into each filler context sample. 
        filler = create_cot_dataset(datasets, tokenizer)
        
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
            
            # Add a datapoint containing all combinations of sample and choice. 
            datapoint = []
            for choice in choices:
                datapoint.append([i for i in itertools.chain.from_iterable(sample + [choice])])

            dataset.append({'choices': datapoint})
    return dataset

def create_cot_random_needle_dataset(args, tokenizer, context_length, document_depth, bias_needle):
    """
    Creates random needle dataset consisting of a single prompt and multiple answer choices. 
    Optionally will add a "biasing needle" to the prompt along with filler context to bias the choices. 
    The needle will be injected somewhere in the filler context based on the document depth parameter.
    """
    # cot_prompt = "Let's think step by step, "
    # Format needle string as needed. 
    if len(bias_needle) > 0:
        bias_needle = tokenizer(bias_needle)['input_ids']
    else: 
        bias_needle = []
    
    # Special case of no context, only a single prompt can be created. 
    if context_length == 0:
        
        # Create dataset with only the input prompt and bias needle if provided. 
        dataset = [{'choices': [bias_needle + tokenizer(args.cot_prompt)['input_ids']]}]
    else:
        
        # First create filler dataset with n_samples each with context_length tokens.
        filler = create_filler_dataset(args, tokenizer, context_length)
    
        # Inject bias needle into each filler context sample. 
        filler = inject_needle(filler, bias_needle, document_depth)
        
        # Tokenize input prompt and choices. 
        input_prompt = tokenizer(args.cot_prompt)['input_ids']
        
        # Create dataset with input prompt and choices, keep track of lengths of filler contexts so we can cache computation over choices. 
        dataset = []
        
        # Final dataset will contain all possible combinations of filler context and choices.
        print('Creating dataset with all samples and choice combinations...')
        for sample in tqdm(filler): 

            # Add input prompt. 
            sample.append(input_prompt)
            
            # Add a datapoint containing all combinations of sample and choice. 
            datapoint = []
            
            datapoint.append([i for i in itertools.chain.from_iterable(sample)])

            dataset.append({'choices': datapoint})
    return dataset

def sample_with_cot(model, tokenizer, inputs, max_length=20, temperature=0.7):
    """
    Perform token-by-token sampling with Chain-of-Thought reasoning.
    Handles batch sampling, generates a sequence for each example in the batch.
    Ensures attention mask and padding are correctly handled.
    """
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']  # Get attention mask from inputs
    batch_size = input_ids.size(0)  # Get the batch size
    eos_token_id = tokenizer.eos_token_id

    # Start generating tokens
    generated_ids = input_ids
    generated_ids = model.generate(input_ids, max_new_tokens=max_length, temperature=temperature)
    generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_ids]
    
    return generated_texts

def reorder_sequence(args, sequence, bias_needle):
    args_needle = bias_needle
    sentence_b= args.cot_prompt
    # Define a regex pattern to match the structure of the sequence
    pattern = re.escape(args_needle) + r'(.*)' + re.escape(sentence_b) + r'(.*)'

    # Use re.sub to swap context after B and context before B
    reordered_sequence = re.sub(pattern, args_needle + ' ' + sentence_b + r'\2\1', sequence)

    return reordered_sequence
    
def random_needle_cot_dataset(args, model, tokenizer, context_len, doc_depth, bias_needle):
    """
    Perform sampling-based generation to get results, handling batched input.
    Generates token-by-token using Chain-of-Thought prompts and compares to choices.
    """
    # Create dataset using existing logic
    dataset = create_cot_random_needle_dataset(args, tokenizer, context_len, doc_depth, bias_needle)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=0)
        
    # Store results for final probabilities
    all_seq_probs = np.zeros((len(dataset), len(args.choices)))
    datasets = []
    for i, batch in enumerate(tqdm(dataloader)):
        inputs = {k: v.cuda() for k, v in batch.items()}
        # Token-by-token sampling, generate final output sequences for the batch
        generated_sequences = sample_with_cot(model, tokenizer, inputs)
        # print(generated_sequences[0])
        datasets.append(generated_sequences[0])
        
    # Return the computed results
    datasets =  [reorder_sequence(args, seq, bias_needle) for seq in datasets]
    
    return datasets

def run_random_needle_exp(args, model, tokenizer, context_len, doc_depth, bias_needle, expected_dist):
    
    # Create dataset. 
    dataset = create_dataset(args, model, tokenizer, context_len, doc_depth, bias_needle)
        
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

def run_point_experiment(args, model, tokenizer, stats, stats_path):
    """
    Run experiment for single data point (i.e. no context, document depth, or biasing needle used.)
    """
    
    # Skip if we've already computed this.
    if 'unconditional_point' in stats:
        return stats
    
    # Otherwise, run experiment and store results.
    stats['unconditional_point'] = run_random_needle_exp(args, model, tokenizer, 0, None, "", preprocess_expected_dist(args.default_expected_dist))
    
    # Save result. 
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
        
def run_pivot_table_experiment(args, model, tokenizer, stats, stats_path, use_needle=True):
    
    # Determine whether to use bias needles or not. 
    if use_needle:
        bias_needles = args.bias_needles
        key = 'needle_pivot_table'
    else: 
        bias_needles = [""]
        key = 'pivot_table'
    
    # Add key to stats if it doesn't exist.
    if key not in stats:
        stats[key] = {}
    
    # Iterate over context lengths, document depths, and bias needles.
    pbar = tqdm(total=len(args.context_lengths) * len(args.document_depths) * len(bias_needles))
    
    for context_len in args.context_lengths: 
        for doc_depth in args.document_depths: 
            for i in range(len(args.bias_needles)):
            
                # Get matching bias needle and expected dist. 
                bias_needle = args.bias_needles[i]
                expected_dist = preprocess_expected_dist(args.needle_expected_dists[i])
            
                # Skip if we've already computed this.
                if (context_len, doc_depth, bias_needle) in stats[key]:
                    pbar.update(1)
                    continue
                
                # Otherwise, run experiment and store results.
                stats[key][(context_len, doc_depth, bias_needle)] = run_random_needle_exp(args, model, tokenizer, context_len, doc_depth, bias_needle, expected_dist)    
                
                # Save in case we get interrupted. 
                with open(stats_path, 'wb') as f:
                    pickle.dump(stats, f)
                    
                pbar.update(1)
                
    # Finish up. 
    pbar.close()


if __name__ == '__main__':
    
    # Arguments. 
    parser = argparse.ArgumentParser('Random Needle Experiments')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('model', type=str, default="google/gemma-2-2b", help='Model to use.')
    parser.add_argument('--use_context', action='store_true', help='Whether to use context and document depth.')
    parser.add_argument('--filler_dataset', type=str, default="Salesforce/wikitext", help="Dataset to use for filler context.")
    parser.add_argument('--dataset_subset', type=str, default="wikitext-103-v1", help="Subset of dataset to use.")
    parser.add_argument('--dataset_split', type=str, default="train", help="Split of dataset to use.")
    parser.add_argument('--n_samples', type=int, default=10, help="Number of samples to use to compute statistics.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    args = parser.parse_args()
    
    # Random seed. 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config file. 
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
        # Update args with config.
        for key, value in config.items():
            setattr(args, key, value)
    
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
    
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Add padding token if needed. 
    if tokenizer.pad_token_id is None: 
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.padding_side = 'left'
    
    # Path to where we'll store stats. 
    results_dir = os.path.join(args.experiment_dir, args.model.replace('/', '_'))
    os.makedirs(results_dir, exist_ok=True)
    stats_path = os.path.join(results_dir, 'stats.pkl')
    
    # Pre-load stats if we already have some results to avoid recomputing.
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
    else: 
        stats = {}
    
    # Generate output for single data point case. 
    print('\n\nRunning point experiment...')
    run_point_experiment(args, model, tokenizer, stats, stats_path)
    gen_point_outputs(args, stats, results_dir)
    
    # Run pivot table experiment without using a bias needle. 
    print('\n\nRunning pivot table experiment...')
    run_pivot_table_experiment(args, model, tokenizer, stats, stats_path, use_needle=False)
    gen_multipoint_outputs(args, stats, results_dir, key='pivot_table')
    
    # Run pivot table experiment and visualize them. 
    print('\n\nRunning bias needle pivot table experiment...')
    run_pivot_table_experiment(args, model, tokenizer, stats, stats_path)        
    gen_multipoint_outputs(args, stats, results_dir, key='needle_pivot_table')