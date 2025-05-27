import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import pdb
import argparse
from tqdm import tqdm
import os
import pickle
import numpy as np

# List of instruct models we'll use. 
from token_prob_experiments import INSTRUCT_MODELS

# List of special characters to insert. 
special_characters = ['*', '^', '#', '@', '$', '%', '&', '!', '~', '<']

# Lambdas to use for experiments. 
lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]

# ICL story context lengths. 
icl_context_lengths = [100, 250, 500]

# ICL story. 
icl_story = open('poisson_icl_story.txt', 'r').read()

def modify_story_with_poisson(special_char, lambda_param, story, n_words=1000):
    """
    Insert a special character into a story string following a poisson distribution with a given lambda parameter.
    """
    # Split the story into words
    words = story.split()
    
    # Cut the story to the first n_words
    words = words[:n_words]
    
    # Initialize the modified story list
    modified_story = []
    
    # Iterate through the words
    for word in words:
        modified_story.append(word)
        
        # Draw a sample from the Poisson distribution
        insert_after_words = np.random.poisson(lambda_param)
        
        # If the sample is greater than 0, insert the special character
        if insert_after_words > 0:
            modified_story.append(special_char)
    
    # Join the list back into a string
    return ' '.join(modified_story)

def compute_lambda_poisson(story, special_str):
    
    # Count the total number of occurrences
    count = story.count(special_str)
    
    # Length of the story (in terms of words) where special_str could have been inserted
    total_words = len(story.replace(special_str, '').split())
    
    # Make sure that we have at least one word in the story
    if total_words == 0:
        return None
    
    # Compute lambda as the average number of occurrences per word
    # or per some other unit of text (e.g., sentences, paragraphs) depending on your modeling preference
    lambda_poisson = count / total_words
    
    return lambda_poisson

def generate_input_prompt(special_str, instruct_model, lambda_param, input_story=None):
    """
    Generate input IDs for a model accounting for whether it's an instruct model or not. 
    """

    # Generate prompt for instruct model. 
    if instruct_model: 
        
        # Define the chat prompt for instruct models
        content = f'Generate a long, multi-paragraph story of your choice. You must insert the "{special_str}" character between words at a frequency following a Poisson distribution with lambda={lambda_param}.'
        
        if input_story: 
            content += f'\nFor example, a story with these insertions could look like the following: "{input_story}".'
        
        prompt = [
            {
                'role': 'user', 
                'content': content
            }
        ]
        
    else: 
        prompt = f'I am going to tell you a long story and insert the "{special_str}" character between words at a frequency following a Poisson distribution with lambda={lambda_param}.\n'
        
        if input_story:
            prompt += f"{input_story}"
        else: 
            prompt += "Once upon a time "
        
    # Generate inputs and place on GPU. 
    return prompt

def generate_input_prompt_icl(instruct_model, story):
    """
    Generate ICL input IDs for a model accounting for whether it's an instruct model or not. 
    """

    # Generate prompt for instruct model. 
    if instruct_model: 
        
        # Define the chat prompt for instruct models
        content = f'Complete the following story: {story}'
        
        prompt = [
            {
                'role': 'user', 
                'content': content
            }
        ]
        
    else: 
        prompt = f'{story}'
        
    # Generate inputs and place on GPU. 
    return prompt

def generate_story_with_special_string(special_str, instruct_model, lambda_p, max_new_tokens, icl_context_length, in_context_learning, pipe, n_samples=10):
    
    # Generate input IDs for model in both standard as well as ICL case. 
    input_story = modify_story_with_poisson(special_str, lambda_p, icl_story, n_words=icl_context_length) if in_context_learning else None
    
    if in_context_learning:
        prompt = generate_input_prompt_icl(instruct_model, input_story)
    else: 
        prompt = generate_input_prompt(special_str, instruct_model, lambda_p, input_story)
    
    # Collect lambdas. 
    lambdas = []

    # Generate N samples worth of lambda estimates. 
    while True: 
        with torch.no_grad():
            
            # Generate story. 
            outputs = pipe(prompt, max_new_tokens=max_new_tokens, temperature=1.0, do_sample=True, return_full_text=False)
            story = outputs[0]['generated_text']

            # Compute lambda from story. 
            story_lambda = compute_lambda_poisson(story, special_str)
            
            # Append to list of lambdas if valid. 
            if story_lambda is not None:
                lambdas.append(story_lambda)
                
            if len(lambdas) == n_samples:
                return lambdas

if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize', action='store_true', help='Quantize the model.')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it', help='Model name.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Token limit.')
    parser.add_argument('--experiment_dir', type=str, default='exps/poisson', help='Path to experiment directory.')
    parser.add_argument('--in_context_learning', action='store_true', help='Use in-context learning.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing stats.')
    args = parser.parse_args()

    # Load the tokenizer and model from HuggingFace
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if args.quantize else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
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

    # Pipeline. 
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

    # Determine if using instruct model or not. 
    instruct_model = args.model in INSTRUCT_MODELS

    # Path to where we'll store stats.
    results_dir = os.path.join(args.experiment_dir, args.model.replace('/', '_'))
    os.makedirs(results_dir, exist_ok=True)
    stats_name = 'stats.pkl' if not args.in_context_learning else 'stats_icl.pkl'
    stats_path = os.path.join(results_dir, stats_name)

    # Pre-load stats if we already have some results to avoid recomputing.
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
    else:
        stats = {}

    # Compute number of trials for progress bar. 
    icl_context_lengths = icl_context_lengths if args.in_context_learning else [None]
    n_trials = len(special_characters) * len(lambdas) * len(icl_context_lengths)
    pbar = tqdm(total=n_trials)

    for special_char in special_characters:
        for lambda_param in lambdas:
            for icl_context_length in icl_context_lengths:
            
                # Skip if we've already computed this. 
                if (special_char, lambda_param, icl_context_length) in stats and not args.overwrite:
                    pbar.update(1)
                    continue
                
                # Otherwise, compute the lambdas. 
                stats[(special_char, lambda_param, icl_context_length)] = generate_story_with_special_string(special_char, instruct_model, lambda_param, args.max_new_tokens, icl_context_length, args.in_context_learning, pipe)    

                # Save and update progress bar. 
                with open(stats_path, 'wb') as f:
                    pickle.dump(stats, f)
                    
                pbar.update(1)