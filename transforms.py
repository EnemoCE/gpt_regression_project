from random import shuffle


class Transform:
    transform_params = None
    auto_transform_params = None

    @classmethod
    def switch_layers(cls, ls):
        switch_params = cls.transform_params.get('switch_params')
        if switch_params:
            i, j = switch_params
            ls[i], ls[j] = ls[j], ls[i]
        return ls

    @classmethod
    def duplicate_layers(cls, ls):
        duplicate_params = cls.transform_params.get('duplicate_params')
        if duplicate_params:
            start, end, repeat = duplicate_params
            new_layers = ls[start-1:end]
            ls = ls[:start-1] + new_layers * repeat + ls[end:]
        return ls
    
    @classmethod
    def slice_layers(cls, ls):
        slice_params = cls.transform_params.get('slice_params')
        if slice_params:
            start, end = slice_params
            ls = ls[start:end + 1]
        return ls


    @classmethod
    def auto_permute_layers(cls, ls):
        permute_bounds_params = cls.auto_transform_params.get('permute_bounds_params')
        if permute_bounds_params:
            start, end = permute_bounds_params
            shuffle(ls[start:end + 1])
        return ls

    @classmethod
    def auto_custom_permute(cls, ls):
        return ls[-3:] + ls[3:-3] + ls[:3]
    


def variant(func_names, variant_type):
    def decorator(func):
        def wrapper(ls):
            for func_name in func_names:
                method_name = f'{func_name}'
                if variant_type == 'auto':
                    assert 'auto' in func_name
                if hasattr(Transform, method_name):
                    method = getattr(Transform, method_name)
                    ls = method(ls)
            return func(ls)
        return wrapper
    return decorator

# Transform functions
def transform_func(ls):
    return ls

def auto_transform_func(ls):
    return ls