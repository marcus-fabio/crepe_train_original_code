import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curves():
    # Replace 'file.tsv' with the path to your actual .tsv file
    file_path = 'experiments/my_test/learning-curve.tsv'

    # Read the .tsv file into a pandas DataFrame
    data = pd.read_csv(file_path, sep='\t')

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data['loss'], label='Loss', marker='o')
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss', marker='o')

    # Add labels, title, and legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Set the x-axis to display integers
    plt.xticks(ticks=data['epoch'], labels=data['epoch'].astype(int))

    # Show the plot
    plt.show()

if __name__ == '__main__':
    plot_learning_curves()