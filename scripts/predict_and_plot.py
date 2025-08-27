import numpy as np
import torch

from solps_ai.predict import load_checkpoint, predict_multi, scale_params
from solps_ai.plotting import plot_te_curvilinear, plot_te_rectilinear  # ok for any field

def main(channel="Te", use_curvilinear=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + normalizer (+ optional param scaler)
    model, norm, (param_mu, param_std) = load_checkpoint("unet_best.pt", device)

    # Load reference mask/coords (+ channel names)
    with np.load("solps_raster_dataset.npz", allow_pickle=True) as d:
        mask_ref = (d["mask"][0] > 0.5).astype(np.float32)   # (H,W)
        Rg = d["Rg"]                                         # (H,W) or (H,W) grid
        Zg = d["Zg"]
        target_keys = d.get("target_keys", None)
        if target_keys is not None:
            # ensure list[str]
            target_keys = [str(k) for k in target_keys.tolist()]
        else:
            target_keys = ["Te"]  # fallback

    # Pick channel index
    if isinstance(channel, str):
        channel = target_keys.index(channel) if channel in target_keys else 0

    # Param vector (raw physical), then scaled if scaler present
    params_raw = [3.0e21, 2.0e6, 1.2e22, 0.35, 0.15]
    params = scale_params(params_raw, param_mu, param_std)

    # Predict ALL channels in physical units, select one
    Y_all = predict_multi(model, norm, mask_ref, params, device=device, as_numpy=True)  # (C,H,W)
    field = Y_all[channel]
    title = f"{target_keys[channel]} (phys units)"

    # Plot (pick one depending on your coordinate convention)
    if use_curvilinear:
        plot_te_curvilinear(field, Rg, Zg, mask=mask_ref, title=title, fname=f"{target_keys[channel]}_RZ.png")
    else:
        plot_te_rectilinear(field, mask=mask_ref, title=title, fname=f"{target_keys[channel]}_XY.png")

if __name__ == "__main__":
    # change channel name to "ne", "ni", "ti", ... as needed
    main(channel="Te", use_curvilinear=True)

