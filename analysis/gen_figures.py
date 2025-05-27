import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np

def gen_point_outputs(args, stats, results_dir):
    """
    Generate experiment output for a single data point (i.e. no context or document depth used.)
    """
    
    # Unpack stats. 
    uncon_point_kl = stats['unconditional_point']['kl_div']
    biased_point_kl = np.mean([point_stats['kl_div'] for point_stats in stats['biased_point'].values()])
    
    # Bar graph for both unconditional and biased point KLs. 
    fig, ax = plt.subplots()
    ax.bar(['Unconditional', 'Biased'], [uncon_point_kl, biased_point_kl])
    ax.set_ylabel('KL Divergence')
    ax.set_title(f'KL Divergence for {args.model}')
    
    # Save figure. 
    save_path = os.path.join(results_dir, 'point_bargraph.png')
    plt.savefig(save_path)

def gen_multipoint_outputs(args, stats, results_dir, key='pivot_table'):
    """
    Generate experiment output for multiple data points (i.e. context and document depth used.)
    
    Pivot table code based on: https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/viz/CreateVizFromLLMTesting.ipynb
    """
    
    # Format data as pandas dataframe.
    data = []
    
    for (context_len, doc_depth, _), point_stats in stats[key].items():
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
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#0CD79F", "#EBB839", "#F0496E"])

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
    save_path = os.path.join(results_dir, f'{key}.png')
    plt.savefig(save_path)