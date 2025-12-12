import torch
from solps_ai.data import make_loaders
from solps_ai.train import train_unet

import numpy as np


#npz_path = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/solps_raster_dataset_te.npz"

npz_path = "/Users/42d/Unet/scripts/solps_raster_dataset_source_particle.npz"

def main():
    


    inputs_mode = "params"  # or "autoencoder"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#    train_loader, val_loader, norm, P, (H,W), param_scaler = make_loaders(
#        npz_path, inputs_mode=inputs_mode, batch_size=16, split=0.85, device=device.type
#    )
    train_loader, val_loader, norm, P, (H,W), param_scaler = make_loaders(
        npz_path, inputs_mode=inputs_mode, batch_size=4, split=0.8, device=device.type
    )

    in_ch = 1 + P if inputs_mode != "autoencoder" else 1
#    model, hist = train_unet(
#        train_loader, val_loader, norm, in_ch, device,
#        inputs_mode=inputs_mode, lam_grad=0.2, lam_w=1.0, lam_ev=0.0,
#        epochs=5, base=32, param_scaler=param_scaler, save_path="unet_best.pt",
#        return_history=True, history_path="training_history.npz"
#    )

    model, hist = train_unet(
        train_loader, val_loader, norm, in_ch, device,
        inputs_mode=inputs_mode, lam_grad=0.2, lam_w=1.0, lam_ev=0.0,
        epochs=2, base=32, param_scaler=param_scaler, save_path="unet_best.pt",
        return_history=True, history_path="training_history.npz"
    )



if __name__ == "__main__":
    main()


