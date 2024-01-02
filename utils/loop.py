import matplotlib.pyplot as plt
import sys
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb

def train(model, iterator, optimizer, criterion, device, run=0):
    
    model.train()
    
    epoch_loss = 0
    acc_temp=0
    # total_sample=0
    
    for _, batch in tqdm(enumerate(iterator)):
        
        optimizer.zero_grad()
        
        src = batch[0].to(device)
        target = batch[1].to(device)
        
        output = model(src)
                    
        loss = criterion(output.squeeze(1), target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # label=target.cpu().numpy()
        # pred=np.argmax(output.detach().cpu().numpy(),axis=1)
        # acc_temp+=np.sum(pred==label)
        # total_sample+=label.shape[0]
        
        if run==1:
            wandb.log({"train/iter_loss" : loss.item()}) 
        

    #accuracy=acc_temp / total_sample
    
    #return epoch_loss / len(iterator), accuracy
    return epoch_loss / len(iterator), np.sqrt(epoch_loss / len(iterator))

def evaluate(model, iterator, criterion, device, run=0):
    
    model.eval()
    epoch_loss = 0
    acc_temp=0
    total_sample=0

    with torch.no_grad():
        for _, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            
            loss = criterion(output.squeeze(1), trg)
            epoch_loss += loss.item()
            
            # label=trg.cpu().numpy()
            # pred=np.argmax(output.detach().cpu().numpy(),axis=1)
            # acc_temp+=np.sum(pred==label)
            # total_sample+=label.shape[0]
            
            if run==1:
                wandb.log({"train/iter_loss" : loss.item()}) 
    
    #accuracy=acc_temp / total_sample
    
    return epoch_loss / len(iterator) , np.sqrt(epoch_loss / len(iterator))# , accuracy

