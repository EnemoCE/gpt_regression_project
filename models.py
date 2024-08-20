from copy import copy, deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F


from transformers import GPT2Model, GPT2Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'




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
        start, end, repeat = self.transform_params.duplicate_params
        new_n_layer = self.configuration.n_layer + (end + 1 - start) * (repeat - 1)
        new_configuration.n_layer = new_n_layer
        return new_configuration


    def recompose(self, model_variant):

        if model_variant == 'modified':
            layers = list(self._backbone.h.children())
            transform_params = self.transform_params
            if transform_params:
                layers = transform_params.transform_func(layers)

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
        elif model_variant == 'modified + no_final_layer_norm':
            new_backbone = deepcopy(self._backbone)
            new_backbone.ln_f = nn.Identity()
        else:
            return None
        return new_backbone.to(device)
    

    def update_new_backbone(self):
        self._h_new_backbone = dict()
        for ele in self.model_variants:
            self._h_new_backbone[ele] = self.recompose(ele)

    def clear_read_out2(self):
        self._read_out2 = self._read_out
    

    def _new_backbone(self, *args, model_variant=None, **kwargs):
        return self._h_new_backbone[model_variant](*args, **kwargs)


    
    def auto_recompose(self):
        layers = list(self._backbone.h.children())
        if self.auto_transform_params:
            layers = self.auto_transform_params.auto_transform_func(layers)
        self._backbone.h = nn.ModuleList(layers)


    def forward(self, xs, targets=None, eval=False, dp=0):
        full_backbone_copy = self.transform_params.full_backbone_copy
        no_layernorm_full_backbone_copy = self.transform_params.no_layernorm_full_backbone_copy
        embeds = self._read_in(xs)
        output = self._backbone(inputs_embeds=embeds, output_attentions=self.output_attentions, output_hidden_states=True)
        if eval:
            with torch.no_grad():
                if self.transform_params.first_n_layers:
                    output_hidden_state = output.hidden_states[self.transform_params.first_n_layers].clone().detach()
                    logits2 = self._read_out(output_hidden_state)
                else:
                    in_output = embeds.clone().detach()
                    if not full_backbone_copy:
                        with torch.set_grad_enabled(self.transform_params.new_backbone_training):
                            in_output = self._new_backbone(inputs_embeds=in_output, model_variant = self.model_variants[0]).last_hidden_state
                            logits2 = self._read_out(in_output)
                    else:
                        with torch.set_grad_enabled(self.transform_params.new_backbone_training):
                            k = 0
                            if no_layernorm_full_backbone_copy == True:
                                k = 1
                            in_output = self._new_backbone(inputs_embeds=in_output, model_variant = self.model_variants[k]).last_hidden_state
                            """
                            with torch.enable_grad():
                                in_output = self._transmit(in_output)
                            """
                            in_output = self._new_backbone(inputs_embeds=in_output, model_variant = self.model_variants[0]).last_hidden_state
            if self.transform_params.readout2_training:
                logits2 = self._read_out2(in_output)
        logits = self._read_out(output.last_hidden_state)
        if targets is None:
            loss = None
            loss2 = None
        else:
            logits = logits[:, ::2, 0]
            B, TC = logits.shape
            logits = logits.view(B*TC)
            targets = targets[:, ::2, 0]
            targets = targets.view(B*TC)
            loss = F.mse_loss(logits, targets)

            if eval:
                logits2 = logits2[:, ::2, 0]
                B2, TC2 = logits2.shape
                logits2 = logits2.view(B2*TC2)
                loss2 = F.mse_loss(logits2, targets)
            else:
                loss2 = None
                logits2 = None

        return logits, logits2, loss, loss2


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