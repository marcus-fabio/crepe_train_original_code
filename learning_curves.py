import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curves():
    file_path = 'experiments/my_test/learning-curve.tsv'

    # Read the .tsv file into a pandas DataFrame
    data = pd.read_csv(file_path, sep='\t')

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data['loss'], label='Loss', marker='.')
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss', marker='.')

    # Add labels, title, and legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_rpa_learning_curves():
    file_path = 'experiments/my_test/learning-curve.tsv'
    rpa_path = 'experiments/my_test/local-average-rpa.tsv'
    # rca_path = 'experiments/my_test/local-average-rca.tsv'

    # Read the .tsv files into pandas DataFrames
    loss_data = pd.read_csv(file_path, sep='\t')
    rpa_data = pd.read_csv(rpa_path, sep='\t')
    # rca_data = pd.read_csv(rca_path, sep='\t')

    # Create the figure and the first axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot loss data on the left y-axis
    ax1.plot(loss_data['epoch'], loss_data['loss'], label='Loss', marker='.', color='b')
    ax1.plot(loss_data['epoch'], loss_data['val_loss'], label='Validation Loss', marker='.', color='g')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    # Create the second axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot RPA data on the right y-axis
    ax2.plot(range(len(rpa_data['mdbsynth'])), rpa_data['mdbsynth'], label='RPA', marker='.', color='r')
    # ax2.plot(range(len(rca_data['mdbsynth'])), rca_data['mdbsynth'], label='RCA', marker='.', color='y')
    ax2.set_ylabel('RPA')
    ax2.tick_params(axis='y')
    ax2.set_yticks(list([tick / 10 for tick in range(0, 11)]))
    ax2.legend(loc='center left')

    # Add title and grid
    plt.title('Learning Curves: Loss and RPA')
    # ax1.grid(True)
    ax2.grid(True)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    # plot_learning_curves()
    plot_rpa_learning_curves()