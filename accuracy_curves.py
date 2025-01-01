import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_curves():
    # Replace 'file.tsv' with the path to your actual .tsv file
    # mae_path = 'experiments/my_test/local-average-mae.tsv'
    rpa_path = 'experiments/my_test/local-average-rpa.tsv'
    # rca_path = 'experiments/my_test/local-average-rca.tsv'


    # Read the .tsv file into a pandas DataFrame
    # mae_data = pd.read_csv(mae_path, sep='\t')
    rpa_data = pd.read_csv(rpa_path, sep='\t')
    # rca_data = pd.read_csv(rca_path, sep='\t')

    # Plot the data
    plt.figure(figsize=(10, 6))
    # plt.plot(range(len(mae_data['mdbsynth'])), mae_data['mdbsynth'], label='MAE', marker='o')
    plt.plot(range(len(rpa_data['mdbsynth'])), rpa_data['mdbsynth'], label='RPA', marker='o')
    # plt.plot(range(len(rca_data['mdbsynth'])), rca_data['mdbsynth'], label='RCA', marker='o')

    # Add labels, title, and legend
    plt.xlabel('Epoch')
    plt.ylabel('%')
    # plt.title('Mean Absolute Error | Raw Pitch Accuracy | Raw Chroma Accuracy')
    plt.title('Raw Pitch Accuracy')
    # plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    plot_accuracy_curves()