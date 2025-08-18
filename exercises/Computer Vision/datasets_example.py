import torch
from torch.utils.data import DataLoader
from torchvision  import transforms
from datasets import load_dataset
from PIL import Image
import io

ds_train = load_dataset("elsaEU/ELSA_D3", split="train", streaming=True)
ds_val = load_dataset("elsaEU/ELSA_D3", split="validation", streaming=True)

ds_train = ds_train.take(800)
ds_val = ds_train.take(200)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

