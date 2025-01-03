import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pandas as pd

def plot_rpa_learning_curves_with_trend():
    file_path = 'experiments/my_test/learning-curve.tsv'
    rpa_path = 'experiments/my_test/local-average-rpa.tsv'
    rca_path = 'experiments/my_test/local-average-rca.tsv'

    # Read the .tsv files into pandas DataFrames
    loss_data = pd.read_csv(file_path, sep='\t')
    rpa_data = pd.read_csv(rpa_path, sep='\t')
    rca_data = pd.read_csv(rca_path, sep='\t')

    # Create the figure and the first axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot loss data
    ax1.plot(loss_data['epoch'], loss_data['loss'], label='Loss', marker='.', color='blue')
    ax1.plot(loss_data['epoch'], loss_data['val_loss'], label='Validation Loss', marker='.', color='green')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    # Create the second axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot RPA and RCA data with trend lines
    for col, data, color in zip(['mdbsynth', 'mdbsynth'], [rpa_data, rca_data], ['red', 'yellow']):
        # Plot the original data
        ax2.plot(range(len(data[col])), data[col], label=col.upper(), marker='.', color=color)

        # Calculate and plot the trend line
        slope, intercept, _, _, _ = linregress(range(len(data[col])), data[col])
        trend = slope * np.arange(len(data[col])) + intercept
        ax2.plot(range(len(data[col])), trend, label=f'{col.upper()} Trend', linestyle='--', color=color)

    ax2.set_ylabel('RPA | RCA')
    ax2.tick_params(axis='y')
    ax2.legend(loc='center left')

    # Add title and grid
    plt.title('Learning Curves: Loss and RPA with Trend Lines')
    ax1.grid(True)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    plot_rpa_learning_curves_with_trend()
