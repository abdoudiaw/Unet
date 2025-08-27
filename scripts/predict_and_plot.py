import numpy as np
import torch
from solps_ai.predict import load_checkpoint, predict_te, scale_params
from solps_ai.plotting import plot_te_curvilinear, plot_te_rectilinear

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + normalizer (+ optional param scaler)
    model, norm, (param_mu, param_std) = load_checkpoint("unet_best.pt", device)

    # Your mask + coordinates
    with np.load("solps_raster_dataset.npz") as d:
        mask_ref = (d["mask"][0] > 0.5).astype(np.float32)    # (H,W)
        # Either you have 1D axes:
        # R_1d, Z_1d = d["R"], d["Z"]
        # Or 2D centers:
        R2d = d["R2d"]; Z2d = d["Z2d"]

    # Param vector (raw physical), scaled if scaler present
    params_raw = [3.0e21, 2.0e6, 1.2e22, 0.35, 0.15]
    params = scale_params(params_raw, param_mu, param_std)

    Te_map_eV = predict_te(model, norm, mask_ref, params, device=device, as_numpy=True)

    # Plot (choose one depending on your coordinates)
    plot_te_curvilinear(Te_map_eV, R2d, Z2d, mask=mask_ref, title="Te (eV)", fname="Te_RZ.png")
    # OR:
    # plot_te_rectilinear(Te_map_eV, R_1d, Z_1d, mask=mask_ref, title="Te (eV)", fname="Te_RZ.png")

if __name__ == "__main__":
    main()


