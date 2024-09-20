from transforms import transform_func as im_transform_func, auto_transform_func as im_auto_transform_func, Transform, variant


class Config:
    def __init__(self, conf_data, transform_func, auto_transform_func):
        self.conf_data = conf_data
        # Dynamically setting attributes from the conf_data
        for key, value in conf_data.items():
            setattr(self, key, value)
        self.experiment_conf.transform_conf['transform_func'] = transform_func
        self.experiment_conf.auto_transform_conf['auto_transform_func'] = auto_transform_func




def build_conf(conf_data):
    #print(f"Running experiment with params:\n{pretty__repr(conf_data.experiment_conf)}")
    Transform.transform_params = conf_data.experiment_conf.transform_conf
    Transform.auto_transform_params = conf_data.experiment_conf.auto_transform_conf


    transform_variants = conf_data.experiment_conf.transform_conf.transform_variants
    transform_choice = conf_data.experiment_conf.transform_conf.transform_choice
    auto_transform_variants = conf_data.experiment_conf.auto_transform_conf.auto_transform_variants
    auto_transform_choice = conf_data.experiment_conf.auto_transform_conf.auto_transform_choice
    transform_func = variant([transform_variants[transform_choice-1]], 'copy')(im_transform_func)
    auto_transform_func = variant([auto_transform_variants[auto_transform_choice-1]], 'auto')(im_auto_transform_func)

    return Config(conf_data, transform_func, auto_transform_func)
