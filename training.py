import wandb
import torch
import numpy as np
import os

from curriculum import Curriculum
from eval import evaluate_test_task, estimate_loss
from task_sampler import get_batch
from plot_utils import built_fig, plt_icl, setup_plot_params




def train_step(model, args, eval=False):
    optimizer = args.optimizer
    batch_size = args.training.batch_size
   

    xb, yb = get_batch(args.curriculum, batch_size)
    # evaluate the loss
    logits, logits2, loss1, loss2 = model(xb, yb, eval)
    optimizer.zero_grad(set_to_none=True)
    loss1.backward()
    if loss2:
        loss2.backward()
    optimizer.step()




def train(model, args):
    args.curriculum = Curriculum(args.training.curriculum)
    curriculum = args.curriculum
    out_dir = args.out_dir
    max_iters = args.training.max_iters
    eval_interval = args.training.eval_interval
    args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate)


    log_model_weights = args.experiment_conf.log_model_weights
    permute_model = args.experiment_conf.auto_transform_conf.permute_model
    permute_interval = args.experiment_conf.auto_transform_conf.permute_interval
    short_description = args.experiment_conf.short_description
    

    wandb.init(
            project = args.wandb.project,
            entity = args.wandb.entity,
            config = args.model.__dict__,
            resume = True,
        )
    
    plot_step, palette, cl_offset = setup_plot_params(args)
    args.plot_step = plot_step
    color_it = label_it = 0

    x_examples = np.arange(1, curriculum.n_points+1)
    fig, axes, labels, hdisplay = built_fig(args)

    for iter in range(max_iters):
        train_step(model, args)
        if iter % eval_interval == 0 or iter == max_iters - 1:
            model.update_new_backbone()
            if model.transform_params.readout2_training:
                for i in range(100):
                    train_step(model, args, eval=True)
            losses = estimate_loss(model, args)
            loss1, loss2 = losses[0], losses[1]
            print(f"step {iter}: train loss {loss1:.4f}, train loss 2 {loss2:.4f}")
            wandb.log(
                {
                    "loss1": loss1,
                    "loss2": loss2,
                },
                step=iter,
            )
            model.clear_read_out2()
        
        if (iter % curriculum.n_points_schedule.interval == 0 or iter == plot_step) and iter != 0:
            x_examples = np.arange(1, curriculum.n_points+1)
        
        if (iter % curriculum.n_dims_schedule.interval == 0  or iter == plot_step) and iter != 0:
            model.update_new_backbone()
            er1_m = max(evaluate_test_task(model, curriculum, 250, logs_ch=1))
            er2_m = max(evaluate_test_task(model, curriculum, 250, logs_ch=2))
            label_it = color_it = 0
            fig_iter = iter
            if iter == plot_step:
                fig_iter = 0
            fig, axes, labels, hdisplay = built_fig(args, height=max(er1_m, er2_m), iter=(fig_iter, fig_iter+curriculum.n_dims_schedule.interval))
            if log_model_weights:
                torch.save(model.state_dict(), os.path.join(out_dir,  'checkpoints', short_description, f"model_{short_description}_{iter}.pt"))

        
        
        #if iter % plot_step == 0 and iter != 0:
            #plt.close(fig)
        
        if permute_model and iter % permute_interval == 0 and iter != 0:
            model.auto_recompose()
        if iter % plot_step == 0 and iter != 0:
            errors1 = evaluate_test_task(model, curriculum, 500, logs_ch=1)
            errors2 = evaluate_test_task(model, curriculum, 500, logs_ch=2)

        
            keys = []
            data1 = [[x, y] for (x, y) in zip(range(1, curriculum.n_points+1), errors1)]
            table1 = wandb.Table(data=data1, columns = ["Number of in-context examples", "Squared error"])
            data2 = [[x, y] for (x, y) in zip(range(1, curriculum.n_points+1), errors2)]
            table2 = wandb.Table(data=data2, columns = ["Number of in-context examples", "Squared error"])


            wandb.log(
            {"my_lineplot_id" : wandb.plot.line(table1, "Number of in-context examples", 
            "Squared error", stroke=None, title="in-context learning 1"),
                    "my_lineplot_id2" : wandb.plot.line(table2, "Number of in-context examples", 
            "Squared error", stroke=None, title="in-context learning 2")}, step=iter)

            
            plt_icl(axes[0], x_examples, errors1, color=palette[color_it+cl_offset], 
                    label = labels[label_it], hdisplay=hdisplay, fig=fig)
            plt_icl(axes[1], x_examples, errors2, color=palette[color_it+cl_offset], 
                    label = labels[label_it], hdisplay=hdisplay, fig=fig)
            fig.savefig(os.path.join(out_dir, 'plots', short_description, f"model_{short_description}_{iter}"))
            color_it += 1
            label_it += 1
    
        curriculum.update()

