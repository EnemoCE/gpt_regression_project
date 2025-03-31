import wandb
import torch
import numpy as np
import os

from curriculum import Curriculum
from eval import evaluate_test_task, evaluate_iterative_newton_test_task, evaluate_test_task_for_errors_distribution, estimate_loss
from task_sampler import get_batch
from plot_utils import built_fig, plt_icl, setup_plot_params
from plot_layer_errors import plot_layer_errors, plot_and_analyze_error_for_normality
from plot_emb_confusion_mx import plot_emb_confusion
from configurations import update_transform
from copy import deepcopy
import torch.nn as nn



def train_step(model, args, base_model=True, iter=None):
    optimizer = args.optimizer
    batch_size = args.training.batch_size
   
    
    xb, yb = get_batch(args.curriculum, batch_size, iter)
    logits, loss =  model(xb, yb, base_model)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    found_nan_inf = False
    for name, param in model.named_parameters():
        if param.grad is not None: # Проверяем и градиенты тоже на всякий случай
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"NaN/Inf found in GRADIENT of {name}")
                found_nan_inf = True
        if torch.isnan(param.data).any() or torch.isinf(param.data).any():
            print(f"NaN/Inf found in WEIGHTS of {name}")
            found_nan_inf = True
    if found_nan_inf:
        print("!!! NaN/Inf detected in parameters/gradients after step !!!")




def train(model, args):
    args.curriculum = Curriculum(args.training.curriculum)
    curriculum = args.curriculum
    out_dir = args.out_dir
    max_iters = args.training.max_iters
    eval_interval = args.training.eval_interval
    args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate)


    log_model_weights = args.experiment_conf.log_model_weights
    show_normality = args.experiment_conf.show_normality
    show_layer_errors = args.experiment_conf.show_layer_errors
    show_embedding_confusion = args.experiment_conf.show_embedding_confusion
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
        train_step(model, args, iter=iter)
        if iter % eval_interval == 0 or iter == max_iters - 1:
            if not model.transform_params.diverge_new_backbone_training:
                model.update_new_backbone()
            if model.transform_params.readout2_training:
                for i in range(eval_interval):
                    train_step(model, args, base_model=False, iter=iter)
            losses = estimate_loss(model, args)
            loss1, loss2 = losses[0], losses[1]
            print(f"step {iter}/{max_iters}: train loss {loss1:.4f}, train loss 2 {loss2:.4f}")
            wandb.log(
                {
                    "loss1": loss1,
                    "loss2": loss2,
                },
                step=iter,
            )
            if model.transform_params.clear_readout2:
                model.clear_readout2()
        
        if (iter % curriculum.n_points_schedule.interval == 0 or iter == plot_step) and iter != 0:
            x_examples = np.arange(1, curriculum.n_points+1)
        
        if (iter % curriculum.n_dims_schedule.interval == 0  or iter == plot_step) and iter != 0:
            if not model.transform_params.diverge_new_backbone_training:
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

    def retrain_readout2(model_i, args, num_steps, layer_number, logs_ch):
        model_i.transform_params.readout2_training = True
        readout2_optimizer = torch.optim.AdamW(model_i._read_out2.parameters(), lr=args.training.learning_rate)
        base_model = True if logs_ch == 1 else False
        for step in range(num_steps):
            xb, yb = get_batch(curriculum, args.training.batch_size)
            logits, loss = model_i(xb, yb, base_model=base_model)
            readout2_optimizer.zero_grad()
            loss.backward()
            if step % 200 == 0:
                print(f"Layer {layer_number} - Step {step}/{num_steps}: Readout2 Training Loss = {loss:.6f}")
            readout2_optimizer.step()

    if not model.transform_params.no_layernorm_full_backbone_copy:

        num_layers = model.new_configuration.n_layer * model.transform_params.full_backbone_rnn_iters
        layer_errors = [[], []]
        layer_errors_newton = []
        layer_numbers = list(range(1, num_layers + 1))
        base_layer_numbers = list(range(1, model.configuration.n_layer + 1))
        layer_embeddings = [[None for j in range(len(base_layer_numbers))] for i in range(len(base_layer_numbers))]


        def get_embdeddings_to_compare(layer_embeddings, layer_nums):
            for i in range(layer_nums):
                for j in range(layer_nums):
                    model_ij = deepcopy(model)
                    model_ij.transform_params.post_eval = True
                    model_ij.transform_params.first_n_layers = i
                    t_v = model_ij.transform_params.transform_variants
                    variant = [i for i in  range(len(t_v)) if t_v[i] == "switch_layers"][0] + 1
                    model_ij.transform_params, model_ij.auto_transform_params = update_transform(model_ij.transform_params,
                                                                                  model_ij.auto_transform_params, variant, new_transform_params=[i+1, j+1])
                    xb, yb = get_batch(curriculum, args.training.batch_size)
                    embedding = model_ij._forward_base_post_eval_hidden(xb)
                    cut_embedding = embedding[:, -1, :]
                    layer_embeddings[i][j] = cut_embedding
            return layer_embeddings


        def post_eval_layers(layer_errors, layer_nums, logs_ch):
            for i in layer_nums:
                model_i = deepcopy(model)
                model_i.transform_params.post_eval = True
                model_i.transform_params.first_n_layers = i
                model_i._read_out2 = nn.Linear(model_i.configuration.n_embd, 1).to(model_i._read_out2.weight.device)
                retrain_readout2(model_i, args, num_steps=model.transform_params.retrain_readout2_iters, layer_number=i, logs_ch=logs_ch)
                errors = evaluate_test_task(model_i, curriculum, count=500, logs_ch=logs_ch)
                error_at_25 = errors[24]
                layer_errors[logs_ch-1].append(error_at_25)
            return layer_errors
        
        def get_errors_for_distribution(layer_num):
            model_ = deepcopy(model)
            model_.transform_params.post_eval = True
            model_.transform_params.first_n_layers = layer_num[-1]
            model_._read_out2 = nn.Linear(model_.configuration.n_embd, 1).to(model_._read_out2.weight.device)
            retrain_readout2(model_, args, num_steps=model.transform_params.retrain_readout2_iters, layer_number=layer_num[-1], logs_ch=1)
            errors = evaluate_test_task_for_errors_distribution(model_, curriculum, count=5000, logs_ch=1)
            return errors

        if show_normality:
            plot_and_analyze_error_for_normality(get_errors_for_distribution(layer_numbers))

        if show_embedding_confusion:
            layer_embeddings = get_embdeddings_to_compare(layer_embeddings, len(base_layer_numbers))
            plot_save_path = os.path.join(out_dir, 'plots', short_description, 'embeddings_similarity.png')
            plot_emb_confusion(layer_embeddings, base_layer_numbers, save_path=plot_save_path)
        
        if show_layer_errors:
            layer_errors = post_eval_layers(layer_errors, base_layer_numbers, logs_ch=1)
            layer_errors = post_eval_layers(layer_errors, layer_numbers, logs_ch=2)
            newton_steps = layer_numbers if len(base_layer_numbers) < len(layer_numbers) else base_layer_numbers

            for i in newton_steps:
                error_newton = evaluate_iterative_newton_test_task(curriculum, num_iterations=i, count=5000)
                layer_errors_newton.append(error_newton)

            plot_save_path = os.path.join(out_dir, 'plots', short_description, 'layer_errors.png')
            plot_layer_errors(layer_numbers, base_layer_numbers, layer_errors, layer_errors_newton, save_path=plot_save_path)
        

    return wandb.run
