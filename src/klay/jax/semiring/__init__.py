from klay.jax.semiring.godel import max_layer, min_layer
from klay.jax.semiring.log import log_sum_layer, encode_input_log
from klay.jax.semiring.real import sum_layer, prod_layer, encode_input_real


def get_semiring(name: str):
    if name == 'real':
        return sum_layer, prod_layer
    elif name == 'log':
        return log_sum_layer, sum_layer
    elif name == 'godel':
        return max_layer, min_layer
    elif name == 'mpe':
        return max_layer, prod_layer
    else:
        raise ValueError(f"Unknown semiring {name}")


def encode_input(name: str):
    if name in ('real', 'godel', 'mpe'):
        return encode_input_real
    elif name == 'log':
        return encode_input_log
    else:
        raise ValueError(f"Unknown semiring {name}")
