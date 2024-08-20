import os
import yaml
import torch
from models import build_model
from training import train



def nested_dict_repr(d, indent=0):
    items = []
    for key, value in d.items():
        if isinstance(value, dict):
            items.append(f"{' ' * indent}{key}:\n{nested_dict_repr(value, indent + 4)}")
        else:
            if isinstance(value, str):
                items.append(f"{' ' * indent}{key}: {value}")
            else:
                items.append(f"{' ' * indent}{key}: {repr(value)}")
    return '\n'.join(items)

def pretty_repr(d):
    return nested_dict_repr(d, indent=4)



def traverse_dict(d, search_key, update_value, found):
    kv = d.items()
    for key, value in kv:
        if key == search_key:
            d[key] = update_value
            found['t'] = 1
            return 1
        elif isinstance(value, dict):
            traverse_dict(d[key], search_key, update_value, found)


def recursive_update_inplace(current_d, updates):
    kv = updates.items()
    found = {'t': 0} 
    for search_key, update_value in kv:
        traverse_dict(current_d, search_key, update_value, found)
        if not found['t']:
            raise ValueError(f'Key {search_key} doesn\'t exist')
    return current_d



class Experiment:
    def __init__(self, conf, short_name = None):
        self.customize_experiment(conf=conf, short_name=short_name)
    
    
    def init_conf_short_description(self, conf, short_name = None):
        self._conf = conf
        if not getattr(self._conf.experiment_conf, 'short_description', None):
            self._conf.experiment_conf.short_description = short_name if short_name else 'baseline'
        
        return self._conf


    def run_experiment(self, conf=None):
        if not conf:
            conf = self._conf
        
        self.experiment_info(conf)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = build_model(conf)
        m = model.to(device)
        weights_path = os.path.join(conf.out_dir, 'checkpoints', conf.experiment_conf.short_description)
        plots_path = os.path.join(conf.out_dir, 'plots', conf.experiment_conf.short_description)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        # Print the number of parameters in the model
        print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')
    
        train(model, conf)

    
    @classmethod
    def yaml_experiment_info(cls, conf=None):
        print(f"Running experiment with params:\n{pretty_repr(conf.experiment_conf)}")

    def experiment_info(self, conf=None):
        if not conf:
            conf = self._conf
        print(f"Running experiment with params:\n{pretty_repr(conf.experiment_conf)}")


    def customize_experiment(self, conf=None, update=None, short_name=None):
        if not conf:
            conf = self._conf
        if update:
            self._conf = recursive_update_inplace(conf.experiment_conf, update)
        conf = self.init_conf_short_description(conf, short_name=short_name)
        return conf
    

