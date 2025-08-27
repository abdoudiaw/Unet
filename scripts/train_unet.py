import torch
from solps_ai.data import make_loaders
from solps_ai.train import train_unet

def main():
    npz_path    = "/Users/42d/Downloads/solps_raster_dataset.npz"
    inputs_mode = "params"  # or "autoencoder"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, norm, P, (H,W), param_scaler = make_loaders(
        npz_path, inputs_mode=inputs_mode, batch_size=16, split=0.85, device=device.type
    )

    in_ch = 1 + P if inputs_mode != "autoencoder" else 1
    _ = train_unet(
        train_loader, val_loader, norm, in_ch, device,
        inputs_mode=inputs_mode, lam_grad=0.2, lam_w=1.0, lam_ev=0.0,
        epochs=10, base=32, param_scaler=param_scaler, save_path="unet_best.pt"
    )

if __name__ == "__main__":
    main()


