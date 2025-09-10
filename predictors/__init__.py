# Import all modules to trigger registration
from . import gaussian
from . import mse
from . import bayescap
from . import generalized_gaussian
from . import evidential
from . import edge_aware_losses
from . import edgy_gaussian

# Keep backward compatibility
from .gaussian import gaussian_nll
from .bayescap import bayescap_loss, predict_bayescap
from .generalized_gaussian import gen_gaussian_nll, predict_gen_gaussian

__all__ = ["gaussian_nll", "predict_gaussian", 
           "gen_gaussian_nll", "predict_gen_gaussian",
           "bayescap_loss", "predict_bayescap"]
