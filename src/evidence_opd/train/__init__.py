def run_training(*args, **kwargs):
    from .trainer import run_training as implementation
    return implementation(*args, **kwargs)
