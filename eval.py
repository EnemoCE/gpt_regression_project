import torch
from task_sampler import get_batch, get_fixed_weight_batch
from models import IteratedNewtonModel
from torchdiffeq import odeint
device = 'cuda' if torch.cuda.is_available() else 'cpu'



@torch.no_grad()
def estimate_loss(model, args):
    curriculum = args.curriculum
    eval_iters = args.training.eval_iters
    batch_size = args.training.batch_size

    out = {}
    model.eval()
    losses = torch.zeros(eval_iters, 2)
    for k in range(eval_iters):
        X, Y = get_batch(curriculum, batch_size)
        logits1, loss1, = model(X, Y)
        logits2, loss2, = model(X, Y, base_model=False)
        losses[k] = torch.tensor([loss1.item(), loss2.item()])
    out = torch.mean(losses, 0)
    model.train()
    return out





@torch.no_grad()
def evaluate_cfm(model, logits):
    B, TC = logits.shape
    x0 = logits.view(B*TC, 1)     # B, TC ->  B*TC, 1
    steps = 200
    t_eval = torch.linspace(0, 1, steps, device=device).type_as(x0)  # (steps)
    features = logits.view(B, 1, TC).expand(B, TC, TC)  # B, TC -> B, TC, TC
    mask = torch.tril(torch.ones(TC, TC), diagonal=-1)  
    mask = mask.to(features.device)
    #mask = mask.unsqueeze(0)  # (1, TC, TC)  not needed due to automatic broadcasting
    features = features * mask  # (B, TC, TC')
    features = features.view(B*TC, -1) #(B*TC, TC')
    def ode_func(t, b_t):
        t = t.view(1, -1).expand(b_t.shape[0], -1).squeeze(1) #(steps) -> (1, steps) -> (B*TC, steps) -> (B*TC*steps)
        return model._cfm_read_out(t, b_t, features)[:,0,:]
    predictions = torch.empty_like(x0).unsqueeze(0)
    num_samples = 30
    for _ in range(num_samples):
        b_pred = odeint(ode_func, x0, t_eval)[-1]
        predictions = torch.cat([predictions, b_pred.detach().unsqueeze(0)], dim=0)
    predictions = predictions.mean(dim=0)
    return predictions.view(B, TC)

@torch.no_grad()
def evaluate_alt_cfm(model, hs_pred, target_features):
    B, T, C = hs_pred.shape 
    hs_pred = hs_pred.view(B*T, C)
    steps = 200 
    t_eval = torch.linspace(0, 1, steps, device=device).type_as(hs_pred)
    h_targets =  model.h_inv(target_features.view(B*T, 1)) # (B*T, C)
    processed_features_hs = model._feature_processor(h_targets.view(B, T, C)) # (B, T, E)

    def ode_func(t, b_t):
        t = t.expand(b_t.shape[0])
        return model._alt_cfm_read_out(t, b_t, processed_features_hs) # (B*T, C)

    predictions = torch.empty_like(hs_pred).unsqueeze(0)
    num_samples = 30
    for _ in range(num_samples):
        b_pred = model._read_out2(odeint(ode_func, hs_pred, t_eval)[-1]).detach()  # (B*T, C) -> (B*T, 1) -> (B*T)
        predictions = torch.cat([predictions, b_pred.unsqueeze(0)], dim=0)
    predictions = predictions.mean(dim=0)
    return predictions.view(B, T)


@torch.no_grad()
def evaluate_uni_cfm(model, back_hs_pred, target_features):
    hs_pred = model._back_to_cfm_read_out(back_hs_pred) #(B, T, E)
    B, T, E = hs_pred.shape 
    steps = 200
    t_eval = torch.linspace(0, 1, steps, device=device).type_as(hs_pred)
    h_targets =  model.h_inv(target_features.view(B*T, 1)).view(B, T, E) # (B, T, E)

    def ode_func(t, b_t):
        t = t.expand(b_t.shape[0])  #(B)
        return model._uni_cfm_read_out(t, b_t) # (B, T, E)

    predictions = torch.empty((B*T), device=hs_pred.device).unsqueeze(0)
    num_samples = 30
    for _ in range(num_samples):
        b_pred = model._read_out2(odeint(ode_func, h_targets, t_eval)[-1]).view(B*T).detach()  # (B, T, E) -> (B, T, 1) -> (B*T)
        predictions = torch.cat([predictions, b_pred.unsqueeze(0)], dim=0)
    predictions = predictions.mean(dim=0)
    return predictions.view(B, T)






@torch.no_grad()
def evaluate_test_task(model, curriculum, count, logs_ch=2):
    model.eval()
    x, targets = get_batch(curriculum, count)
    base_model = True if logs_ch == 1 else False
    logits, loss = model(x, targets=None, base_model=base_model)

    targets = targets[:, ::2, 0]
    if model.transform_params.cfm_loss[0] and logs_ch==2:
        if model.transform_params.cfm_loss[1] == 2:
            logits = evaluate_alt_cfm(model, logits, x[:,1::2,0])
        elif model.transform_params.cfm_loss[1] == 3:
            logits = evaluate_uni_cfm(model, logits, x[:,1::2,0])
        else:
            logits = evaluate_cfm(model, logits, x[:,1::2,0])
    final_errors = torch.square(targets-logits).mean(dim=0)
    model.train()
    return final_errors.tolist()


@torch.no_grad()
def evaluate_test_task_for_errors_distribution(model, curriculum, count, logs_ch=2):
    model.eval()
    x, targets = get_fixed_weight_batch(curriculum, count)
    base_model = True if logs_ch == 1 else False
    logits, loss= model(x, targets, base_model=base_model)
    targets = targets[:, ::2, 0]
    B, TC = targets.shape
    final_errors = targets-logits
    model.train()
    return final_errors.reshape(TC, B).tolist()



@torch.no_grad()
def evaluate_iterative_newton_test_task(curriculum, num_iterations, count):

    input_dim = curriculum.n_dims_truncated
    newton_model = IteratedNewtonModel(input_dim=input_dim, num_iterations=num_iterations)
    newton_model.to(device)
    x_train, y_train = get_batch(curriculum, batch_size=count)

    A = x_train[:, 1::2, :]
    y = y_train[:, ::2, 0]

    w_k = newton_model(A, y)

    x25 = A[:, -1, :]
    y25_pred = torch.einsum('ij,ij->i', x25, w_k)
    y25_true = y[:, -1]
    error25 = torch.square(y25_true - y25_pred).mean(dim=0).item()
    return error25
