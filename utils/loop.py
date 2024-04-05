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
    rmse = 0
    #acc_temp=0
    total_sample=0
    
    for _, batch in tqdm(enumerate(iterator)):
        
        optimizer.zero_grad()
        
        src = batch[0].to(device)
        target = batch[1].to(device)
        
        output = model(src)
                    
        loss = criterion(output.squeeze(1), target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),3)
        optimizer.step()

        epoch_loss += loss.item() 
        rmse += loss.item() * target.shape[0]
        total_sample += target.shape[0]
        
        
        if run==1:
            wandb.log({"train/iter_loss" : loss.item()}) 
        

    return epoch_loss / len(iterator), np.sqrt(rmse / total_sample)

def evaluate(model, iterator, criterion, device, run=0):
    
    model.eval()
    epoch_loss = 0
    rmse = 0
    total_sample=0
    
    with torch.no_grad():
        for _, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            
            loss = criterion(output.squeeze(1), trg)
            epoch_loss += loss.item()
            rmse += loss.item() * trg.shape[0]
            total_sample += trg.shape[0]

            if run==1:
                wandb.log({"valid/iter_loss" : loss.item()}) 
    
    return epoch_loss / len(iterator) , np.sqrt(rmse / total_sample)



def train_weighted(model, iterator, optimizer, criterion, device, run=0):
    
    model.train()
    
    epoch_loss = 0
    rmse = 0
    #acc_temp=0
    total_sample=0
    
    for _, batch in tqdm(enumerate(iterator)):
        
        optimizer.zero_grad()
        
        src = batch[0].to(device)
        mask = batch[1].to(device).squeeze(1)
        target = batch[2].to(device)
        
        output = model(src)
                    
        loss = criterion(output.squeeze(1), target, mask)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),3)
        optimizer.step()

        epoch_loss += loss.item() 
        rmse += loss.item() * target.shape[0]
        total_sample += target.shape[0]
        
        
        if run==1:
            wandb.log({"train/iter_loss" : loss.item()}) 
        

    return epoch_loss / len(iterator), np.sqrt(rmse / total_sample)

def evaluate_weighted(model, iterator, criterion, device, run=0):
    
    model.eval()
    epoch_loss = 0
    rmse = 0
    total_sample=0
    
    with torch.no_grad():
        for _, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[2].to(device)
            output = model(src)
            
            loss = criterion(output.squeeze(1), trg)
            epoch_loss += loss.item()
            rmse += loss.item() * trg.shape[0]
            total_sample += trg.shape[0]

            if run==1:
                wandb.log({"valid/iter_loss" : loss.item()}) 
    
    return epoch_loss / len(iterator) , np.sqrt(rmse / total_sample)

def train_quantile(model, iterator, optimizer, criterion, device, run=0):
    
    model.train()    
    epoch_loss = 0
    rmse = 0
    #acc_temp=0
    total_sample=0
    
    for _, batch in tqdm(enumerate(iterator)):
        
        optimizer.zero_grad()
        
        src = batch[0].to(device)
        target = batch[1].to(device)
        
        output = model(src)
                    
        loss = criterion(output, target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),3)
        optimizer.step()

        epoch_loss += loss.item() 
        rmse += loss.item() * target.shape[0]
        total_sample += target.shape[0]
        
        
        if run==1:
            wandb.log({"train/iter_loss" : loss.item()}) 
        

    return epoch_loss / len(iterator), np.sqrt(rmse / total_sample)

def evaluate_quantile(model, iterator, criterion, device, run=0):
    
    model.eval()
    epoch_loss = 0
    rmse = 0
    total_sample=0
    
    with torch.no_grad():
        for _, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            rmse += loss.item() * trg.shape[0]
            total_sample += trg.shape[0]

            if run==1:
                wandb.log({"valid/iter_loss" : loss.item()}) 
    
    return epoch_loss / len(iterator) , np.sqrt(rmse / total_sample)
