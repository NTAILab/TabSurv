import string
from datetime import datetime
import os
import random

def avoid_singleton_batch(
    sample_size: int,
    batch_size: int,
    use_normalization: bool,
) -> int:
    """Avoid a final one-row batch, which is invalid for BatchNorm training."""
    if use_normalization and sample_size % batch_size == 1:
        return batch_size + 1
    return batch_size

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string

def get_checkpoint_name(model_name:str, rnd_len: int) -> str:
    now = datetime.now()
    time_string = now.strftime('%Y_%m_%d_%H_%M_%S')
    pid = os.getpid()
    return f"{model_name}_{generate_random_string(rnd_len)}_{time_string}_{pid}"
