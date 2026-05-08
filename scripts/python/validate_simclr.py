import os
import argparse
import torch
import numpy as np
import h5py
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

class SimpleH5Dataset(Dataset):
    def __init__(self, h5_path, img_size=224):
        self.h5_path = h5_path
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        with h5py.File(self.h5_path, 'r') as f:
            self.keys = list(f['images'].keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            img = f['images'][self.keys[idx]][()]
        img = Image.fromarray(img).convert('L')
        return self.transform(img)

def get_features(model, loader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            feat = model(imgs)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

def main():
    parser = argparse.ArgumentParser(description="SimCLR Feature Validation")
    parser.add_argument("--data_h5", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", default="./val_results")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model with ResNet50 backbone
    model = models.resnet50(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Identity()
    
    # Load weights
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    dataset = SimpleH5Dataset(args.data_h5)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    features = get_features(model, loader, device)
    np.save(os.path.join(args.outdir, "features.npy"), features)
    print(f"Validation complete. Features saved to {args.outdir}")

if __name__ == "__main__":
    main()
