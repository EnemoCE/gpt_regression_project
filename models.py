from copy import copy, deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F


from transformers import GPT2Model, GPT2Config
from configurations import update_transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class IteratedNewtonModel(nn.Module):
    def __init__(self, input_dim, num_iterations):
        super(IteratedNewtonModel, self).__init__()
        self.input_dim = input_dim
        self.num_iterations = num_iterations
        self.device = device
        self.S = None
        self.X = None

    def initialize(self, A):
        self.S = A.transpose(1,2) @ A 
        SS_T = self.S @ self.S.transpose(1, 2)
        eigenvalues = torch.linalg.eigvalsh(SS_T)
        lambda_max = torch.max(eigenvalues).item()
        alpha = 2 / lambda_max
        self.X = alpha * self.S

    def iterate(self, A):
        for step in range(self.num_iterations):
            self.X = 2 * self.X - self.X @ self.S @ self.X

    def forward(self, A, y):
        if self.X is None:
            self.initialize(A)
        self.iterate(A)
        self.w = (self.X @ A.transpose(1, 2) @ y.unsqueeze(2))
        return self.w.squeeze(2)





class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, transform_params=None, auto_transform_params = None, 
                 model_variants = ['basic'], output_attentions=False):
        super(TransformerModel, self).__init__()
        self.configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.transform_params = transform_params
        self.auto_transform_params = auto_transform_params
        self.model_variants = model_variants
        self.output_attentions = output_attentions
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(self.configuration)
        self.new_configuration = self.build_new_config()
        self.update_new_backbone()
        self._read_out = nn.Linear(n_embd, 1)
        self._read_out2 = nn.Linear(n_embd, 1)
        self._transmit = nn.Linear(n_embd, n_embd)


    def build_new_config(self):
        new_configuration = deepcopy(self.configuration)
        new_n_layer = self.configuration.n_layer

        if self.transform_params.duplicate_params:
            start, end, repeat = self.transform_params.duplicate_params
            new_n_layer = self.configuration.n_layer + (end + 1 - start) * (repeat - 1)
        elif self.transform_params.slice_params:
            start, end= self.transform_params.slice_params
            new_n_layer = end + 1 - start

        new_configuration.n_layer = new_n_layer
        return new_configuration


    def base_backbone_updater(self, layers):
        new_backbone = self._backbone.__class__(self.new_configuration)
        check_out = [id(ele) for ele in self._backbone.h.parameters()]
        with torch.no_grad():
            for paramA, paramB in zip(new_backbone.parameters(), self._backbone.parameters()):
                if id(paramB) not in check_out:
                    paramA.copy_(paramB)
        with torch.no_grad():
            for i, layer in enumerate(layers):
                for paramA, paramB in zip(new_backbone.h[i].parameters(), layer.parameters()):
                    paramA.copy_(paramB)
        
        new_backbone.h = nn.ModuleList(layers)
        return new_backbone
        

    def recompose(self, model_variant):

        if model_variant == 'modified':
            layers = list(self._backbone.h.children())
            transform_params = self.transform_params
            if transform_params:
                layers = transform_params.transform_func(layers)
            new_backbone = self.base_backbone_updater(layers)
        elif model_variant == 'full_backbone + no_final_layer_norm':
            new_backbone = deepcopy(self._backbone)
            new_backbone.ln_f = nn.Identity()
        else:
            return None
        return new_backbone.to(device)



    def update_new_backbone(self):
        self._h_new_backbone = dict()
        for ele in self.model_variants:
            self._h_new_backbone[ele] = self.recompose(ele)


    def clear_readout2(self):
        self._read_out2 = self._read_out


    def _new_backbone(self, *args, model_variant=None, **kwargs):
        return self._h_new_backbone[model_variant](*args, **kwargs)


    def auto_recompose(self, hard_update=False):
        layers = list(self._backbone.h.children())
        if self.auto_transform_params:
            layers = self.auto_transform_params.auto_transform_func(layers)
        if hard_update:
            self._backbone = self.base_backbone_updater(layers).to(device)
            return
        self._backbone.h = nn.ModuleList(layers)
    
    
    @torch.no_grad()
    def collect_embedding(self, xs):
        #self.update_new_backbone()
        self.auto_recompose()
        #variant = [i for i in  range(t_v) if t_v[i] == "slice_layers"][0] + 1
        #self.transform_params, self.auto_transform_params = update_transform(self.transform_params, self.auto_transform_params, variant)
        #self.auto_recompose()
        return self._forward_base_post_eval_hidden(xs)


    @torch.no_grad()
    def _forward_base_post_eval(self, embeds):
        first_n_layers = self.transform_params.first_n_layers
        in_output = embeds.clone().detach()
        logits = None
        in_output = self._backbone(inputs_embeds=in_output, output_hidden_states=True).hidden_states[first_n_layers]
        with torch.set_grad_enabled(True):
            logits =  self._read_out2(in_output)
        return logits
    
    @torch.no_grad()
    def _forward_base_post_eval_hidden(self, xs, targets=None, base_model=True):
        embeds = self._read_in(xs)
        first_n_layers = self.transform_params.first_n_layers
        in_output = embeds.clone().detach()
        return self._backbone(inputs_embeds=in_output, output_hidden_states=True).hidden_states[first_n_layers]

    
    def _forward_modified(self, embeds):
        no_layernorm_full_backbone_copy = self.transform_params.no_layernorm_full_backbone_copy
        first_n_layers = self.transform_params.first_n_layers
        full_backbone_rnn_iters = self.transform_params.full_backbone_rnn_iters
        in_output = embeds.clone().detach()

        k = 1 if no_layernorm_full_backbone_copy else 0
        logits2 = None

        with torch.set_grad_enabled(self.transform_params.new_backbone_training):
            if not first_n_layers:
                first_n_layers =  self.new_configuration.n_layer * full_backbone_rnn_iters

            for i in range(full_backbone_rnn_iters):
                full_backbone_count =  self.new_configuration.n_layer * (i+1)
                if full_backbone_count > first_n_layers:
                    current_n_layers =  first_n_layers % (self.new_configuration.n_layer * i) if i else first_n_layers
                    in_output = self._new_backbone(inputs_embeds=in_output, model_variant = self.model_variants[k],
                                                    output_hidden_states=True).hidden_states[current_n_layers]
                    break
                in_output = self._new_backbone(inputs_embeds=in_output, model_variant = self.model_variants[k]).last_hidden_state

        if self.transform_params.readout2_training:
            logits2 = self._read_out2(in_output)
        else:
            logits2 = self._read_out(in_output)
        
        return logits2



    def forward(self, xs, targets=None, base_model=True):

        embeds = self._read_in(xs)
        output = self._backbone(inputs_embeds=embeds, output_attentions=self.output_attentions, output_hidden_states=True)
        logits = None
    
        if base_model and not self.transform_params.post_eval:
            logits = self._read_out(output.last_hidden_state)
        elif base_model:
            logits = self._forward_base_post_eval(embeds)
        else:
            logits = self._forward_modified(embeds)


        if targets is None:
            loss = None
        else:
            logits = logits[:, ::2, 0]
            B, TC = logits.shape
            logits = logits.view(B*TC)
            targets = targets[:, ::2, 0]
            targets = targets.view(B*TC)
            loss = F.mse_loss(logits, targets)

        return logits, loss


def build_model(conf):
    if conf.model.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.model.n_dims,
            n_positions=conf.model.n_positions,
            n_embd=conf.model.n_embd,
            n_layer=conf.model.n_layer,
            n_head=conf.model.n_head,
            model_variants = conf.experiment_conf.transform_conf.model_variants,
            transform_params = conf.experiment_conf.transform_conf,
            auto_transform_params = conf.experiment_conf.auto_transform_conf
        )
    else:
        raise NotImplementedError
    return model
