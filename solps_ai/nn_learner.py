import torch
from torch import nn

class ParamToZ(nn.Module):
    def __init__(self, P, latent_dim, hidden=(256,256), dropout=0.0):
        super().__init__()
        layers = []
        d = P
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, p):
        return self.net(p)

def train_param2z(P_train, Z_train, P_val, Z_val, device="cuda", epochs=200, lr=1e-3, batch=256):
    P_train = torch.tensor(P_train, dtype=torch.float32)
    Z_train = torch.tensor(Z_train, dtype=torch.float32)
    P_val   = torch.tensor(P_val,   dtype=torch.float32)
    Z_val   = torch.tensor(Z_val,   dtype=torch.float32)

    Pdim = P_train.shape[1]
    Zdim = Z_train.shape[1]

    model = ParamToZ(Pdim, Zdim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(P_train))
        for i in range(0, len(P_train), batch):
            idx = perm[i:i+batch]
            p = P_train[idx].to(device)
            z = Z_train[idx].to(device)
            pred = model(p)
            loss = loss_fn(pred, z)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep % 10 == 0 or ep == epochs-1:
            model.eval()
            with torch.no_grad():
                pv = P_val.to(device)
                zv = Z_val.to(device)
                lv = loss_fn(model(pv), zv).item()
            print(f"ep {ep:04d}  val_mse={lv:.4e}")

    return model
