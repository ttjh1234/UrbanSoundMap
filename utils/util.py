import numpy as np
import time
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

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
    

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    
def load_teacher(model_name, model_dict, model_path, n_cls):
    print('==> loading teacher model')
    model = model_dict[model_name](num_classes=n_cls)
    
    try:
        print("Single GPU Model Load")
        model.load_state_dict(torch.load(model_path))
        print("Load Single Model")
    except:
        print("Mutil GPU Model Load")
        state_dict=torch.load(model_path)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.','')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        print("Load Single GPU Model from Multi GPU Model")
    
    print('==> done')
    return model