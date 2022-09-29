from torch.utils.tensorboard import SummaryWriter

import timm
import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from torch.utils.tensorboard import SummaryWriter


import sys
sys.path.append('../utils')

from data_loader import DataLoader_iNaturalist
from tqdm import tqdm
import numpy as np 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def transform():
    return transforms.Compose([
    Resize((500,500)),
    ToTensor()])


def main(model):
    print(model)
    
    try:
        model = timm.create_model(model, pretrained=True, num_classes = 4271).to(device)
    except Exception as e:
        print(e)

    train_loader = DataLoader_iNaturalist(
        '/home/team_050/data_2021_mini/2021_train_mini', 
        transform = transform(),
        target_level = 'species'
    )

    val_loader = DataLoader_iNaturalist(
        '/home/team_050/data_validation/2021_valid/', 
        transform = transform(),
        target_level = 'species'
    )

    loader = DataLoader(train_loader, batch_size = 32, shuffle = True, num_workers=4)
    loader_val = DataLoader(val_loader, batch_size = 32, shuffle = True, num_workers=4)


    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter()
    for epoch in range(8):
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = loss_func(pred, labels)
            writer.add_scalar("Loss/train", loss, epoch)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

        #validate each epoch
        with torch.no_grad():
            val_loss, size = 0,0
            for images, labels in loader_val:
                    images = images.to(device)
                    labels = labels.to(device)
                    pred = model(images)
                    loss = loss_func(pred, labels)
                    val_loss += int(loss.item())
                    size += len(labels)
            writer.add_scalar("Loss/val", val_loss / size, epoch)
            
    torch.save(model, 'final_model.pkl')

if __name__ == '__main__':
    import sys
    model = sys.argv[1]
    main(model)
