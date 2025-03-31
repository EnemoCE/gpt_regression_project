import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline
import scipy.stats as stats
import numpy as np



def plot_and_analyze_error_for_normality(data):
    for i in range(len(data)):
        sns.histplot(data[i], kde=True)
        stats.probplot(data[i], dist="norm", plot=plt)
        plt.show()
        stat, p = stats.normaltest(data[i])
        print(f"Статистика: {stat}, p-значение: {p}  Тест Д’Агостино–Пирсона")
        mu, sigma = np.mean(data[i]), np.std(data[i], ddof=1)  # Оценка параметров нормального распределения
        stat, p = stats.kstest(data[i], 'norm', args=(mu, sigma))
        print(f"Статистика: {stat}, p-значение: {p} Тест Колмогорова-Смирнова")

def plot_layer_errors(layer_numbers, base_layer_numbers, layer_errors, newton_errors, save_path=None):
    sns.set(style="whitegrid", rc={"grid.linewidth": 0.5})

    num_layers = len(layer_numbers)
    width = min(14, num_layers * 1)
    plt.figure(figsize=(width, 6))
    
    smooth_layer_errors = [[], []]
    smooth_newton_errors = []

    newton_steps_numbers = layer_numbers if len(base_layer_numbers) < len(layer_numbers) else base_layer_numbers

    x_smooth = np.linspace(1, layer_numbers[-1], 300)
    xb_smooth = np.linspace(1, base_layer_numbers[-1], 300)
    xn_smooth = np.linspace(1, newton_steps_numbers[-1], 300)

    spl1 = make_interp_spline(base_layer_numbers, layer_errors[0], k=3)
    smooth_layer_errors[0] = spl1(xb_smooth)

    spl2 = make_interp_spline(layer_numbers, layer_errors[1], k=3)
    smooth_layer_errors[1] = spl2(x_smooth)

    spl3 = make_interp_spline(newton_steps_numbers, newton_errors, k=3)
    smooth_newton_errors = spl3(xn_smooth)


    plt.plot(
        xb_smooth,
        smooth_layer_errors[0],
        linestyle='--',
        color='plum',
        linewidth=1.5,
        label='Base model'
    )


    plt.plot(
        x_smooth,
        smooth_layer_errors[1],
        linestyle='--',
        color='mediumpurple',
        linewidth=1.5,
        label='Modified model'
    )


    plt.plot(
        xn_smooth,
        smooth_newton_errors, 
        linestyle='-', 
        color='red',
        linewidth=1,
        label='Iterative Newton\'s Method',
        alpha=0.15
    )

    
    plt.scatter(
        base_layer_numbers,
        layer_errors[0],
        marker='h',
        s=30,
        color='plum',
    )


    plt.scatter(
        layer_numbers,
        layer_errors[1],
        marker=(5,1),
        s=30,
        color='mediumpurple'
    )

    plt.xlabel('Transformer Layer Number / Newton Steps', fontsize=12)
    plt.ylabel('Squared Error', fontsize=12)
    plt.title('Error at 25th Example vs Transformer Layers and Newton Steps', fontsize=14)

    plt.xlabel('Transformer Layer Number (first_n_layers)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Error at 25th Example vs Transformer Layer', fontsize=14)

    plt.legend(loc='lower right', fancybox=True, shadow=True, ncol=1, bbox_to_anchor=(0.85, 0.06))

    plt.yscale('log')

    def y_formatter(x, pos):
        exponent = int(np.log10(x))
        return f"$10^{{{exponent}}}$"

    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_formatter))
    plt.gca().set_yticks([1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
