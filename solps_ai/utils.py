import numpy as np

def scale_params_for_inference(params, param_mu, param_std):
    params = np.asarray(params, dtype=np.float32)
    if param_mu is not None and param_std is not None:
        params = (params - np.asarray(param_mu, dtype=np.float32)) / np.asarray(param_std, dtype=np.float32)
    return params

