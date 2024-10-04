import math
import numpy as np
import seaborn as sns
from IPython import display

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sns.set(palette='summer')
plt.ioff()

def setup_plot_params(conf):
    curriculum = conf.curriculum

    #color_count = 4*max_iters//curriculum.n_dims_schedule.interval
    graphs_per_plot = min(5, curriculum.n_dims_schedule.interval//400)

    plot_step = int(math.ceil((curriculum.n_dims_schedule.interval//graphs_per_plot)/100.0)*100)


    color_count = curriculum.n_dims_schedule.interval//400
    palette = sns.color_palette("rocket_r", n_colors=int(color_count*1.1))
    cl_offset = int(color_count*1.1) - color_count
    return plot_step, palette, cl_offset



def plt_icl(ax, x, y, hdisplay, color, label, fig):
    ax.plot(x, y, color=color, label=label, linewidth=0.6)
    ax.legend()
    hdisplay.update(fig)


def y_formatter(x, pos): 
    exponent = int(np.log10(x)) 
    return f"$10^{{{exponent}}}$" 

def built_fig(conf, height=1.5, iter=None):
    transform_conf = conf.experiment_conf.transform_conf
    curriculum = conf.curriculum
    plot_step = conf.plot_step
    if not iter:
        iter = (0, curriculum.n_dims_schedule.interval)
    fig, axes = plt.subplots(1, 2, figsize=(12, 2*height/1.5))
    fig.suptitle(f"ICL on steps {iter[0]} - {iter[1]}       dims truncated: {curriculum.n_dims_truncated}       number of points: {curriculum.n_points}", fontsize=12)
    hdisplay = display.display("", display_id=True)
    
    logarithmic_scale = conf.experiment_conf.logarithmic_scale
    if logarithmic_scale:
        y_ticks = [1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    x_ticks = np.arange(0, curriculum.n_points+2+(curriculum.n_points) % 2, 2)
    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_xlabel('Number of in-context examples')
        ax.set_ylabel('Squared error')
        ax.set_ylim(0, height+0.1)
        if logarithmic_scale:
            ax.set_ylim(1e-5, height+0.1)  # Adjust y-limits for log scale
            ax.set_yscale('log')  # Set y-axis to log scale
            ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
            ax.set_yticks(y_ticks)

    axes[0].title.set_text('Base model')
    rnn_iters = transform_conf.full_backbone_rnn_iters
    add_string = f' with {rnn_iters} RNN backbone repeats'
    if transform_conf.duplicate_params:
        times_form = 'time' if  transform_conf.duplicate_params[2] == 1 else 'times'
        axes[1].title.set_text(f"Model with {transform_conf.duplicate_params[2]}  {times_form} repeated layers\
        {transform_conf.duplicate_params[0]} - {transform_conf.duplicate_params[1]}")
    elif transform_conf.slice_params:
        axes[1].title.set_text(f"Model with sliced layers {transform_conf.slice_params[0]} - {transform_conf.slice_params[1]} {' and' + add_string if rnn_iters > 1 else ''}")
    else:
        axes[1].title.set_text(f"Modified model \
        {add_string if rnn_iters > 1 else ''}")
    
    start_jump = 0
    if iter[0] == 0:
        start_jump = plot_step
    labels=np.arange(iter[0]+start_jump, iter[1]+start_jump, plot_step)
    
    return fig, axes, labels, hdisplay