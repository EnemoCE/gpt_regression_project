import matplotlib.pyplot as plt
import seaborn as sns

def plot_layer_errors(layer_numbers, layer_errors, save_path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    plt.plot(layer_numbers, layer_errors, marker='o', linestyle='-')

    plt.xlabel('Transformer Layer Number (first_n_layers)')
    plt.ylabel('Error')
    plt.title('Error at 25th Example vs Transformer Layer')

    plt.yscale('log')
    plt.gca().set_yticks([1e0, 1e-1, 1e-2, 1e-3, 1e-4])
    plt.gca().get_yaxis().set_major_formatter(plt.ScalarFormatter())
    plt.gca().get_yaxis().set_minor_formatter(plt.NullFormatter())

    if save_path:
        plt.savefig(save_path)
    plt.show()
