import h5py, time, numpy as np

def scale_params_for_inference(params, param_mu, param_std):
    params = np.asarray(params, dtype=np.float32)
    if param_mu is not None and param_std is not None:
        params = (params - np.asarray(param_mu, dtype=np.float32)) / np.asarray(param_std, dtype=np.float32)
    return params

def save_geometry_h5(path, r2d, z2d, case_name=None, units="m", level=4):
    r = np.ascontiguousarray(np.asarray(r2d, dtype=np.float32))
    z = np.ascontiguousarray(np.asarray(z2d, dtype=np.float32))
    H, W = r.shape
    chunk = (min(512, H), min(512, W))
    with h5py.File(path, "w") as f:
        for name, arr in [("R2D", r), ("Z2D", z)]:
            d = f.create_dataset(name, data=arr, compression="gzip",
                                 compression_opts=int(level), shuffle=True, chunks=chunk)
            d.attrs["units"] = units
            d.attrs["grid"] = "cell centers"
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if case_name is not None: f.attrs["case_name"] = str(case_name)

def load_geometry_h5(path):
    with h5py.File(path, "r") as f:
        return f["R2D"][:], f["Z2D"][:]
