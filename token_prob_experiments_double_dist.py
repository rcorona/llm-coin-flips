from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import spacy
import hashlib

import argparse
from tqdm import tqdm
import itertools
import os
import numpy as np
import yaml
import pickle
import pdb
from functools import wraps
from typing import Any, Dict, List, Tuple, Union
import glob

# Local imports.
from analysis.gen_figures import gen_point_outputs, gen_multipoint_outputs

# List of instruct models.
INSTRUCT_MODELS = [
    'google/gemma-2-2b-it',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.1-70B-Instruct',
    'meta-llama/Llama-3.1-405B-Instruct',
    'microsoft/Phi-3.5-mini-instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'allenai/OLMoE-1B-7B-0924-Instruct'
]

# Collate function for dataloader.
def collate_fn(batch):

    # Unpack batch.
    samples = [item['choices']['input_ids'] for item in batch]

    # Collapse examples into batch of single strings.
    samples = {'input_ids': list(itertools.chain.from_iterable(samples))}

    # Tokenize.
    inputs = tokenizer.pad(samples, return_tensors='pt', padding=True)

    # Choice start idxs. 
    choice_start_idxs = list(itertools.chain.from_iterable([item['choice_start_idx'] for item in batch]))

    # Add padding to choice start idxs.
    padding_ammount = (inputs['attention_mask'] == 0).long().sum(dim=-1)

    for i in range(len(choice_start_idxs)):
        choice_start_idxs[i] += padding_ammount[i].item()

    inputs['choice_start_idxs'] = choice_start_idxs

    return inputs

# def create_filler_dataset(args, tokenizer, context_length):
#     """
#     Preprocesses filter dataset into sentences and then uses them to populate
#     a dataset with the desired number of examples for statistical analysis.
#     """

#     # Will contain lists of tokenized sentences for each sample.
#     samples = []
#     curr_len = 0
#     curr_sample = []

#     # Load sentencizer from Spacy.
#     nlp = spacy.load("en_core_web_sm")

#     # Load dataset.
#     dataset = datasets.load_dataset(path=args.filler_dataset, data_dir=args.dataset_subset)[args.dataset_split]
#     dataset = dataset.shuffle(args.seed) # Fixed seed ensures that we get same context samples for all experiments.

#     # Iterate over dataset and tokenize sentences.
#     for example in dataset:

#         # Split example into sentences.
#         doc = nlp(example['text'])

#         # Tokenize each sentence.
#         for sent in doc.sents:

#             # Tokenize sentence.
#             tokens = tokenizer(sent.text)['input_ids']

#             # Add to current sample.
#             curr_sample.append((tokens, sent.text))
#             curr_len += len(tokens)

#             # Switch to new sample if we've reached the desired length.
#             if curr_len >= context_length:
#                 samples.append(curr_sample)
#                 curr_sample = []
#                 curr_len = 0

#                 # Return if we've reached the desired number of samples.
#                 if len(samples) == args.n_samples:
#                     return samples

#     # We should never reach this point.
#     raise ValueError('Not enough samples in filler dataset to reach desired number of samples!')

# def inject_needle(dataset, needle, document_depth):
#     """
#     Injects a biasing needle into the filler context at a specified depth.
#     """

#     # Inject needle into each sample.
#     for sample in dataset:

#         # Get total length of document.
#         doc_len = sum([len(tokens) for tokens, _ in sample])

#         # Compute number of tokens that would get us to desired depth.
#         token_depth = int(doc_len * document_depth)

#         # Determine which document to inject needle after.
#         inject_idx = 0
#         len_so_far = 0

#         # Find the sentence to inject needle after.
#         for tokens, sent in sample:

#             # Update length so far.
#             len_so_far += len(tokens)
#             inject_idx += 1

#             # If we've reached the desired depth, inject needle after this sentence.
#             if len_so_far >= token_depth:
#                 sample.insert(inject_idx, (None, needle))
#                 break

#     return dataset

def structure_chat(args, instruct_model, i=0):
    """
    Construct input prompt based on model type.
    """

    # Construct prompt for instruct models.
    if instruct_model:
        chat = {'user': args.instruct_prompt[int(i/10)], 'assistant': args.input_prompt[i]}
    else:
        chat = {'user': args.input_prompt[i]}

    return chat

# def tokenize_prompt_and_map_choice_idxs(prompt, tokenizer, choice_start_idx, choice_end_idx):
#     """
#     Tokenize a string pertaining to a full input sequence to LLM. 
#     Given a choice string (e.g. "heads", "tails"), return the token indexes which correspond to the choice.
#     The indexes allow us to extract the logit values for the choice from the model output.
    
#     INPUTS: 
#     prompt -- str: The full input sequence to the LLM in string space. 
#     tokenizer -- transformers.PreTrainedTokenizer: Tokenizer object for the LLM.
#     choice_start_idx -- int: The starting index of the choice string in the prompt. (e.g. index of 'h' in 'heads')
#     choice_end_idx -- int: The ending index of the choice string in the prompt. (e.g. index of 's' in 'heads')
    
#     OUTPUTS: 
#     tokenization -- transformers.tokenization_utils_base.BatchEncoding: Tokenized input sequence.
#     choice_token_idxs -- list: List of token indexes which correspond to the choice string characters. 
#     """
    
#     # Tokenizer prompt. 
#     tokenization = tokenizer(prompt)
    
#     # Get span within prompt for each token. 
#     token_spans = [tokenization.token_to_chars(i) for i in range(len(tokenization['input_ids']))]
    
#     # Will contain token idxs for choice within prompt. 
#     choice_token_idxs = []
    
#     # Will contain vocab idxs for choice tokens.
#     vocab_idxs = []
    
#     # Iterate over tokens and find the ones that correspond to the choice.
#     for idx, span in enumerate(token_spans):
        
#         # Skip if span is None.
#         if span is None:
#             continue
        
#         # Check if token represents part of the choice
#         partial_end = choice_end_idx <= span.end and choice_end_idx > span.start
#         partial_start = choice_start_idx >= span.start and choice_start_idx < span.end

#         if partial_start or partial_end:
#             choice_token_idxs.append(idx)
#             vocab_idxs.append(tokenization['input_ids'][idx])
            
#     return tokenization['input_ids'], choice_token_idxs, vocab_idxs

def find_largest_common_subsequence(sequences):
    """
    Find largest common subsequence between two tokenized strings. 
    """
    
    # Extract tokens. 
    tokenizations = [seq for seq in sequences['input_ids']]
    
    # Find largest common subsequence.
    i = 0
    
    while all(tokenizations[0][i] == tokenization[i] for tokenization in tokenizations):
        i += 1
        
    return i

def choice_sanity_check(choices, choice_start_idx, tokenizer, gt_choices):
    """
    Make sure we've actually found the beginning of the choices for a set of tokenized sequences.
    
    choices -- transformers.tokenization_utils_base.BatchEncoding: Tokenized input sequences.
    choice_start_idx -- int: Index where the choices start in each of the tokenized sequences. 
    tokenizer -- transformers.PreTrainedTokenizer: Tokenizer object for the LLM.
    gt_choices -- list: List of ground truth choice strings.
    """
    
    # Extract choice tokens. 
    choice_tokens = [tokenizer.decode(choices['input_ids'][i][choice_start_idx:]) for i in range(len(choices['input_ids']))]
    
    # Post-process them. 
    choice_strs = [choice.strip() for choice in choice_tokens]
    
    for choice in gt_choices: 
        if choice not in choice_strs:
            return False
        
    return True

# def contextless_dataset(args, tokenizer, chat, bias_needle, instruct_model):
#     """
#     Create contextless dataset for both instruct models and not instruct models.
#     """

#     # Apply chat template to every choice.
#     if instruct_model:

#         choices_str = []

#         for idx, choice in enumerate(args.choices):
#             messages = [
#                 {'role': 'user', 'content': ' '.join((f'{bias_needle} {chat["user"]}').split())}, 
#                 {'role': 'assistant', 'content': ' '.join((chat['assistant'] + choice).split())}
#             ]

#             # Tokenize with chat template.
#             choices_str.append(tokenizer.apply_chat_template(messages, tokenize=False))
            
#         choices = tokenizer(choices_str)

#     # Create dataset with only the input prompt and bias needle if provided.
#     else:

#         # Construct prompt. 
#         print([' '.join((f'{bias_needle} {args.input_prompt}{choice}').split()) for choice in args.choices])
#         choices = tokenizer([' '.join((f'{bias_needle} {args.input_prompt}{choice}').split()) for choice in args.choices])
#         print("")
        
#     # Get index of divergence from largest common subsequence. 
#     choice_start_idx = find_largest_common_subsequence(choices)
    
#     # Sanity check that we've actually found the beginning of the choices. 
#     assert choice_sanity_check(choices, choice_start_idx, tokenizer, args.choices), "Choice start index is incorrect!"
    
#     dataset = [{
#         'choices': choices,
#         'choice_start_idx': [choice_start_idx] * len(args.choices),
#     }]

#     return dataset

# def construct_context_choice(context, choice, tokenizer, chat, instruct_model):
#     """
#     Construct a single tokenized choice sentence based on model type and context.
#     """

#     # Apply chat template.
#     if instruct_model:
#         messages = [
#             {'role': 'user', 'content': ' '.join((f'{context} {chat["user"]}').split())}, 
#             {'role': 'assistant', 'content': ' '.join((chat['assistant'] + choice).split())}
#         ]
        
#         return tokenizer.apply_chat_template(messages, tokenize=False)

#     # Otherwise, just tokenize the context + prompt + choice.
#     else:
#         return f'{context} {chat["user"]}' + choice

# def create_random_needle_dataset(args, tokenizer, context_length, document_depth, bias_needle, instruct_model):
#     """
#     Creates random needle dataset consisting of a single prompt and multiple answer choices.
#     Optionally will add a "biasing needle" to the prompt along with filler context to bias the choices.
#     The needle will be injected somewhere in the filler context based on the document depth parameter.
#     """

#     # Check to see if the dataset is cached
#     hash_key = str(hashlib.md5(str((args.model, args.filler_dataset, args.dataset_subset, args.dataset_split, args.n_samples, context_length, document_depth, bias_needle, args.choices, args.instruct_prompt, args.input_prompt, args.seed)).encode()).hexdigest())
#     if not args.overwrite:
#         if os.path.exists(os.path.join(args.cache_dir, f"{hash_key}.pkl")):
#             with open(os.path.join(args.cache_dir, f"{hash_key}.pkl"), 'rb') as f:
#                 return pickle.load(f)

#     # Construct input prompt depending on whether model is instruct or not.
#     chat = structure_chat(args, instruct_model)

#     # Special case of no context, only a single prompt can be created.
#     if context_length == 0:
#         dataset = contextless_dataset(args, tokenizer, chat, bias_needle, instruct_model)
#     else:

#         # First create filler dataset with n_samples each with context_length tokens.
#         filler = create_filler_dataset(args, tokenizer, context_length)

#         # Inject bias needle into each filler context sample.
#         if len(bias_needle) > 0:
#             filler = inject_needle(filler, bias_needle, document_depth)

#         # Create dataset with input prompt and choices, keep track of lengths of filler contexts so we can cache computation over choices.
#         dataset = []

#         # Final dataset will contain all possible combinations of filler context and choices.
#         for sample in filler:

#             # Extract sentences from sample.
#             context = ' '.join([sent for _, sent in sample]).strip()

#             # Add a datapoint containing all combinations of sample and choice.
#             choices = []

#             for idx, choice in enumerate(args.choices):
#                 choices.append(construct_context_choice(context, choice, tokenizer, chat, instruct_model))

#             choices = tokenizer(choices)

#             # Get index of divergence from largest common subsequence. 
#             choice_start_idx = find_largest_common_subsequence(choices)
            
#             # Sanity check that we've actually found the beginning of the choices. 
#             assert choice_sanity_check(choices, choice_start_idx, tokenizer, args.choices), "Choice start index is incorrect!"

#             dataset.append({'choices': choices, 'choice_start_idx': [choice_start_idx] * len(args.choices)})

#     # Cache dataset.
#     os.makedirs(args.cache_dir, exist_ok=True)
#     with open(os.path.join(args.cache_dir, f"{hash_key}.pkl"), 'wb') as f:
#         pickle.dump(dataset, f)

#     return dataset

def create_double_dist_dataset(args, tokenizer, context_length, document_depth, bias_needle, instruct_model, i):
    """
    Create contextless dataset for both instruct models and not instruct models.
    """

    # TODO: instruct models
    chat = structure_chat(args, instruct_model, i)
    # print(chat)

    # Apply chat template to every choice.
    if instruct_model:

        choices_str = []

        for idx, choice in enumerate(args.choices):
            messages = [
                {'role': 'user', 'content': ' '.join((f'{bias_needle} {chat["user"]}').split())}, 
                {'role': 'assistant', 'content': ' '.join((chat['assistant'] + choice).split())}
            ]

            # Tokenize with chat template.
            choices_str.append(tokenizer.apply_chat_template(messages, tokenize=False))
        
        # print(choices_str)
        choices = tokenizer(choices_str)

    # Create dataset with only the input prompt and bias needle if provided.
    else:

        # Construct prompt. 
        # print([' '.join((f'{bias_needle} {args.input_prompt[i]}{choice}').split()) for choice in args.choices])
        choices = tokenizer([' '.join((f'{bias_needle} {args.input_prompt[i]}{choice}').split()) for choice in args.choices])
        # print("")
        
    # Get index of divergence from largest common subsequence. 
    choice_start_idx = find_largest_common_subsequence(choices)
    
    # Sanity check that we've actually found the beginning of the choices. 
    assert choice_sanity_check(choices, choice_start_idx, tokenizer, args.choices), "Choice start index is incorrect!"
    
    dataset = [{
        'choices': choices,
        'choice_start_idx': [choice_start_idx] * len(args.choices),
    }]

    return dataset


def run_random_needle_exp(args, model, tokenizer, context_len, doc_depth, bias_needle, expected_dist, instruct_model, i=0):

    # Create dataset.
    dataset = create_double_dist_dataset(args, tokenizer, context_len, doc_depth, bias_needle, instruct_model, i)
    # print(dataset)
    # Create dataloader.
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=0)

    # Collect sequence probabilities.
    n_choices = len(args.choices)
    all_seq_probs = np.zeros((len(dataset), n_choices))
    kl_divergences = np.zeros((len(dataset), n_choices))
    attention_maps = [] # Collect attention maps and associated tokens for each input. 

    # Compute probabilities.
    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):

            # Put on gpu.
            all_inputs = {
                'input_ids': batch['input_ids'].cuda(),
                'attention_mask': batch['attention_mask'].cuda(),
            }

            # # Put on cpu.
            # all_inputs = {
            #     'input_ids': batch['input_ids'],
            #     'attention_mask': batch['attention_mask'],
            # }
            
            # Collect logits for all choices.
            all_log_probs = []
            first_token_probs = []

            for i in range(all_inputs['input_ids'].shape[0]):
                
                # Process inputs.
                inputs = {k: v[i].unsqueeze(0) for k, v in all_inputs.items()}
                
                # Get model outputs. 
                outputs = model(**inputs, output_attentions=True)

                ### COLLECT ATTENTION SCORES ###
                # Concatenate attention matrices across layers.
                attention_mats = torch.cat(outputs.attentions).float().cpu().numpy()
                
                # Get associated input tokens. 
                input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                
                # Get logit distribution for token locations pertaining to choice, we shift by one to get the logits for the choice tokens conditioned on prior tokens.
                choice_start_idx = batch['choice_start_idxs'][i] # This is the idx where the choice starts (e.g. where "heads" starts)
                
                # Add to attention maps. 
                attention_maps.append({
                    'attention': attention_mats,
                    'tokens': input_tokens,
                    'choice_start_idx': choice_start_idx
                })
                ### END COLLECT ATTENTION SCORES ###
                
                # Get the logits out of the model output. [Batch, Sequence, Vocab]
                # Gather the time steps pertaining to just the choice tokens. 
                """
                For MDP this will just be the last time step on its own, something like: outputs[:, -1, :] 
                We also don't need to shift by one for MDP since we're not looking for the probability of the current state, rather the next state.
                """ 
                logits = outputs.logits[0, choice_start_idx-1:-1] 
                
                # Compute probability for these tokens over the vocabulary in log space. 
                all_token_log_probs = F.log_softmax(logits, dim=-1)
                
                # Get the vocab IDs for the choice tokens so we can get the right logit values.  
                choice_tokens = inputs['input_ids'][0, choice_start_idx:] # For MDP we want these to be the indices of the state tokens.
                
                # Use the vocab IDs to get the logit values for the choice tokens.
                # e.g. for MDP should be something like [Batch, N_States]
                token_log_probs = all_token_log_probs[torch.arange(all_token_log_probs.size(0)), choice_tokens]  
                
                # Add log prob of choice's first token to list. 
                first_token_probs.append(token_log_probs[0])
                
                # Compute probability for choice by taking product of its token probabilities. 
                choice_log_prob = token_log_probs.sum()
                all_log_probs.append(choice_log_prob)
                
            ### COMPUTE SUM OF FIRST TOKEN LOG PROBABILITIES ###
            first_token_agg_log_prob = torch.logsumexp(torch.stack(first_token_probs), dim=0)
            ### END COMPUTE SUM OF FIRST TOKEN LOG PROBABILITIES ###
                
            # Concatenate outputs.
            all_log_probs = torch.stack(all_log_probs)

            # Uncollapse so we get log probabilities for each answer choice sequence per dataset example.
            seq_log_probs = all_log_probs.view(-1, n_choices)

            # Normalize to get a distribution.
            norm_seq_probs = torch.exp(seq_log_probs - torch.logsumexp(seq_log_probs, dim=1, keepdim=True).expand_as(seq_log_probs))

            # Compute point-wise KL divergence.
            target_dist = expected_dist.view(1, -1).repeat(norm_seq_probs.size(0), 1)
            # kl_div = F.kl_div(torch.log(norm_seq_probs), target_dist, reduction='none', log_target=True)

            # Collect results.
            start = b_idx * args.batch_size
            end = start + norm_seq_probs.size(0)
            all_seq_probs[start:end] = norm_seq_probs.cpu().float().numpy()
            # kl_divergences[start:end] = kl_div.cpu().numpy()

    # Marginalize out over samples.
    choice_probs = np.mean(all_seq_probs, axis=0)

    # Compute KL divergence from expected distribution.
    # kl_div = np.sum(kl_divergences) / kl_divergences.shape[0]

    # Consolidate stats and return them, along with attention maps.
    stats = {
        'choice_probs': choice_probs,
        # 'kl_div': kl_div,
        'all_seq_probs': all_seq_probs,
        'expected_dist': expected_dist.cpu().numpy(),
        'first_token_agg_log_prob': first_token_agg_log_prob.float().cpu().numpy()
    }
    
    # print(choice_probs)
    return stats, attention_maps

def preprocess_expected_dist(expected_dist):
    """
    Convert a list expected distribution into a numpy array in logspace.
    """
    expected_dist = np.asarray(expected_dist).astype(np.float32)

    # Normalize expected dist and convert to tensor in logspace.
    expected_dist = (expected_dist / expected_dist.sum()) + 1e-8
    # expected_dist = torch.log(torch.Tensor(expected_dist).float().cuda())
    expected_dist = torch.log(torch.Tensor(expected_dist).float())

    return expected_dist

def run_point_experiments(args, model, tokenizer, stats, stats_path, instruct_model, exp_dir, bias_1, bias_2, cutoff, icl_len):
    """
    Run experiment for single data point (i.e. no context, document depth, or biasing needle used.)
    """
    print(f"starting exp {bias_1} {bias_2} {cutoff} {icl_len}")

    all_probs = []
    for i in range(len(args.input_prompt)):
        stats, attention_maps = run_random_needle_exp(args, model, tokenizer, 0, None, "", preprocess_expected_dist(args.default_expected_dist), instruct_model, i)
        # we only care about stats["choice_probs"] --> we want a final csv with bias_1, bias_2, cutoff, icl_len, choice_probs, model
        all_probs.append(stats["choice_probs"])

    # print(len(all_probs), all_probs)
    all_p_heads = np.array([prob[0] for prob in all_probs])
    avg_p_heads = np.mean(all_p_heads)
    var_p_heads = np.var(all_p_heads)
    # print(avg_p_heads, var_p_heads)

    stats = {}
    stats["avg_p_heads"] = avg_p_heads
    stats["var_p_heads"] = var_p_heads
    stats["bias_1"] = bias_1
    stats["bias_2"] = bias_2
    stats["cutoff"] = cutoff
    stats["icl_len"] = icl_len

    # # Collect stat for process without bias needle and without context.
    # if 'unconditional_point' not in stats or args.overwrite:
    #     stats['unconditional_point'], attn_maps = run_random_needle_exp(args, model, tokenizer, 0, None, "", preprocess_expected_dist(args.default_expected_dist), instruct_model)

    # Save results.
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # attn_maps_path = os.path.join(exp_dir, 'attn_maps_unconditional_point.pkl')
    # with open(attn_maps_path, 'wb') as f:
    #     pickle.dump(attn_maps, f)

    # # Collect stat for process without context, averaged across bias needles.
    # if 'biased_point' not in stats:
    #     stats['biased_point'] = {}

    # print('Running biased point experiments...')

    # for i in range(len(args.bias_needles)):
    #     if args.bias_needles[i] not in stats['biased_point'] or args.overwrite:
    #         stats['biased_point'][args.bias_needles[i]], attn_maps = run_random_needle_exp(args, model, tokenizer, 0, None, args.bias_needles[i], preprocess_expected_dist(args.needle_expected_dists[i]), instruct_model)

    #         # Save results.
    #         with open(stats_path, 'wb') as f:
    #             pickle.dump(stats, f)
                
    #         attn_maps_path = os.path.join(exp_dir, 'attn_maps_biased_point.pkl')
    #         with open(attn_maps_path, 'wb') as f:
    #             pickle.dump(attn_maps, f)

    # print('Finished biased point experiments...')

# def run_pivot_table_experiment(args, model, tokenizer, stats, stats_path, instruct_model, use_needle=True):

#     # Determine whether to use bias needles or not.
#     if use_needle:
#         bias_needles = args.bias_needles
#         expected_dists = args.needle_expected_dists
#         doc_depths = args.document_depths
#         key = 'needle_pivot_table'
#     else:
#         bias_needles = [""]
#         expected_dists = [args.default_expected_dist]
#         doc_depths = [0]
#         key = 'pivot_table'

#     # Add key to stats if it doesn't exist.
#     if key not in stats:
#         stats[key] = {}

#     # If using gpt-2 need to remove any context window length >= 1000.
#     if 'gpt2' in args.model:
#         args.context_lengths = [length for length in args.context_lengths if length < 1000]

#     # Iterate over context lengths, document depths, and bias needles.
#     pbar = tqdm(total=len(args.context_lengths) * len(doc_depths) * len(bias_needles))

#     for context_len in args.context_lengths:
#         for doc_depth in doc_depths:
#             for i in range(len(bias_needles)):

#                 # Get matching bias needle and expected dist.
#                 bias_needle = bias_needles[i]
#                 expected_dist = preprocess_expected_dist(expected_dists[i])

#                 # Skip if we've already computed this.
#                 if (context_len, doc_depth, bias_needle) in stats[key] and not args.overwrite:
#                     pbar.update(1)
#                     continue

#                 # Otherwise, run experiment and store results.
#                 try:
#                     stats[key][(context_len, doc_depth, bias_needle)], _ = run_random_needle_exp(args, model, tokenizer, context_len, doc_depth, bias_needle, expected_dist, instruct_model)
#                 except Exception as e:
#                     print('Error occurred during experiment.', e)
#                     pass # TODO handle this better, OOM errors can occur.

#                 # Save in case we get interrupted.
#                 with open(stats_path, 'wb') as f:
#                     pickle.dump(stats, f)

#                 pbar.update(1)

#     # Finish up.
#     pbar.close()

if __name__ == '__main__':

    # Arguments.
    parser = argparse.ArgumentParser('Random Needle Experiments')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('model', type=str, default="google/gemma-2-2b", help='Model to use.')
    parser.add_argument('--filler_dataset', type=str, default="Salesforce/wikitext", help="Dataset to use for filler context.")
    parser.add_argument('--dataset_subset', type=str, default="wikitext-103-v1", help="Subset of dataset to use.")
    parser.add_argument('--dataset_split', type=str, default="train", help="Split of dataset to use.")
    parser.add_argument('--n_samples', type=int, default=10, help="Number of samples to use to compute statistics.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results.')
    parser.add_argument('--quantize', action='store_true', help='Quantize model as NF4.')
    parser.add_argument('--cache_dir', type=str, default='./cache/', help='Directory to store cached datasets.')

    args = parser.parse_args()

    # Random seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model.
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # ) if args.quantize else None
    bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        # device_map=None,
        attn_implementation="eager", # TODO add inference-time modification to allow flash-attention.
        torch_dtype=torch.bfloat16, # NOTE: On unsupported GPUs you'll get emulated bfloat16 support instead of true bfloat16.
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Add padding token if needed.
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.padding_side = 'left'

    # # Create experiment directory if it doesn't exist.
    # os.makedirs(os.path.join(args.experiment_dir), exist_ok=True)

    # Path to where we'll store stats.
    # results_dir = os.path.join(args.experiment_dir, args.model.replace('/', '_'))
    icl_len = 100
    results_dir = f"/home/ericwang0533/random_needles/random_needles/exps/contextual/coin_flip_icl_double_dist/{args.model.replace('/', '_')}/icl_{icl_len}"
    os.makedirs(results_dir, exist_ok=True)

    # Pre-load stats if we already have some results to avoid recomputing.
    # if os.path.exists(stats_path):
    #     with open(stats_path, 'rb') as f:
    #         stats = pickle.load(f)
    # else:
    #     stats = {}
    stats = {}

    # Check whether model is in list of instruct models.
    instruct_model = args.model in INSTRUCT_MODELS

    # TODO: figure out instruct model 
    # # Load chat template from file if using an instruct model. 
    if instruct_model: 
        
        # Load the same chat template for all llama instruct models. 
        if 'llama' in args.model:
            chat_template_path = os.path.join('chat_templates', 'meta-llama_Llama-3.1-8B-Instruct.jinja2')
        else:     
            chat_template_path = os.path.join('chat_templates', f'{args.model.replace("/", "_")}.jinja2')
            
        with open(chat_template_path, 'r') as f:
            tokenizer.chat_template = f.read()

    # # Determine which bias needles to use.
    # if instruct_model:
    #     args.bias_needles = args.instruct_bias_needles

    config_file_paths = glob.glob(
        f"/home/ericwang0533/random_needles/random_needles/configs/exps/contextual/coin_flip_icl_double_dist/icl_{icl_len}/**/bias_*.yaml", recursive=True
    )

    for config_file in config_file_paths:
        # Load config file.
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

            # Update args with config.
            for key, value in config.items():
                setattr(args, key, value)

        print("")
        print(config_file)
        bias_folder = config_file.split(".yaml")[0].split("/")[-1]
        cutoff_folder = config_file.split(".yaml")[0].split("/")[-2]
        bias_1 = bias_folder.split("_")[1]
        bias_2 = bias_folder.split("_")[2]
        cutoff = cutoff_folder.split("_")[1]

        if bias_1 != bias_2:
            continue

        results_subdir = os.path.join(results_dir, cutoff_folder, bias_folder)
        os.makedirs(results_subdir, exist_ok=True)

        stats_path = os.path.join(results_subdir, 'stats.pkl')

        if os.path.exists(stats_path):
            print(f"{stats_path} exists")
            continue

        # # Generate output for single data point case.
        print(f'\n\nRunning point experiment...')
        run_point_experiments(args, model, tokenizer, stats, stats_path, instruct_model, results_subdir, bias_1, bias_2, cutoff, icl_len)
        # gen_point_outputs(args, stats, results_subdir)

        # break

    # # Run pivot table experiment without using a bias needle.
    # print('\n\nRunning pivot table experiment...')
    # run_pivot_table_experiment(args, model, tokenizer, stats, stats_path, instruct_model, use_needle=False)
    # gen_multipoint_outputs(args, stats, results_dir, key='pivot_table')

    # # Run pivot table experiment and visualize them.
    # print('\n\nRunning bias needle pivot table experiment...')
    # run_pivot_table_experiment(args, model, tokenizer, stats, stats_path, instruct_model)
    # gen_multipoint_outputs(args, stats, results_dir, key='needle_pivot_table')
