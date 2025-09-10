"""
Registry for predictors and criteria.
"""
PREDICTOR_REGISTRY = {}
CRITERION_REGISTRY = {}

def register_predictor(name):
    """Decorator to register a predictor function."""
    def decorator(func):
        PREDICTOR_REGISTRY[name] = func
        return func
    return decorator

def register_criterion(name):
    """Decorator to register a criterion function."""
    def decorator(func):
        CRITERION_REGISTRY[name] = func
        return func
    return decorator
