import torch
import numpy as np

def train_torch(data_loader, model, num_iterations=100, lr=0.001, optimizer='SGD', gpu=False):
    # Create a Binary-Cross-Entropy (BCE) loss module
    criterion = torch.nn.CrossEntropyLoss()
    # Create an optimizer
    optimizer = getattr(torch.optim,optimizer)(model.parameters(),lr=lr)
    # Now we run the training!
    loss_v=[]
    while num_iterations > 0:
        for data,label in data_loader:
            
            if gpu:
                data,label = data.cuda(),label.cuda()
            # Prediction
            prediction = model(data)
            # Compute loss
            loss = criterion(prediction, label)
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Record loss
            loss_v.append(loss.item())
            # Brake if we consumed all iteration counts
            num_iterations -= 1
            if num_iterations < 1:
                break
        
    return np.array(loss_v)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def run_test(model,loader,gpu=False):

    from scipy.special import softmax

    prediction_v = []
    label_v      = []
    softmax_v    = []
    model.eval()
    
    with torch.set_grad_enabled(False):
        idx=0
        for data,label in loader:
            if gpu:
                data,label = data.cuda(), label.cuda()
            prediction   = model(data).cpu().numpy()
            prediction_v.append ( np.argmax(prediction,axis=1)    )
            label_v.append      ( label.cpu().numpy().reshape(-1) )
            s = softmax(prediction,axis=1)
            softmax_v.append    (s)
            idx +=1
    return np.concatenate(prediction_v), np.concatenate(label_v), np.concatenate(softmax_v)