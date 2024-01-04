import numpy as np
import torch
import time

def rescale_data(x,mean=None,std=None):
    
    '''
    Input x is (C,H,W) image and normalized vec. 
    Output is (H,W,C) image and denormalized vec.
    '''
    
    if not mean:
        mean=np.array((0.9113)).reshape(-1,1,1)
    
    if not std:
        std=np.array((0.2168)).reshape(-1,1,1)
    
    x=x*std+mean
    x=np.transpose(x,[1,2,0])
    return x

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def adjust_learning_rate(epoch, opt, optimizer):
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            
def check_inference_time_for_gpu(model, device):
    dummy_input = torch.zeros((1,1,100,100))
    model.eval()
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    start_event = torch.cuda.Event(enable_timing = True)
    end_event = torch.cuda.Event(enable_timing = True)
    
    with torch.no_grad(): 
        start_event.record()
        _ = model(dummy_input)
        end_event.record()
    
    torch.cuda.synchronize()
    
    time_taken = start_event.elapsed_time(end_event)
    print(f"Elapsed time on GPU : {time_taken * 1e-3} seconds")
    

def check_inference_time_for_cpu(model):
    dummy_input = torch.zeros((1,1,100,100))
    model.eval()
    
    with torch.no_grad(): 
        start = time.perf_counter()
        _ = model(dummy_input)
        end = time.perf_counter()
    
    print(f"Elapsed time on CPU : {end - start} seconds")
    

