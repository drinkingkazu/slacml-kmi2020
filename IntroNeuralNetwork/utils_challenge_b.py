from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from sklearn.cluster import DBSCAN
import matplotlib
import time
from sklearn.metrics import log_loss
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

def save_state(blob, prefix='./snapshot'):
    # Output file name
    filename = '%s-%07d.ckpt' % (prefix, blob.iteration)
    # Save parameters
    # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
    # 2) network weight
    torch.save({
        'global_step': blob.iteration,
        'optimizer': blob.optimizer.state_dict(),
        'state_dict': blob.net.state_dict()
        }, filename)
    return filename

def restore_state(blob):
    # Open a file in read-binary mode
    with open(blob.weight_file, 'rb') as f:
        # torch interprets the file, then we can access using string keys
        checkpoint = torch.load(f)
        # load network weights
        blob.net.load_state_dict(checkpoint['state_dict'], strict=False)
        # if optimizer is provided, load the state of the optimizer
        if blob.optimizer is not None:
            blob.optimizer.load_state_dict(checkpoint['optimizer'])
        # load iteration count
        blob.iteration = checkpoint['global_step']
        

# Plot a confusion matrix
def plot_confusion_matrix(labels,prediction,class_names):
    """
    Args:
          prediction ... 1D array of predictions, the length = sample size
          class_names ... 1D array of string label for classification targets, the length = number of categories
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,8),facecolor='w')
    num_labels = len(class_names)
    mat,_,_,im = ax.hist2d(labels,prediction,
                           bins=(num_labels,num_labels),
                           range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(class_names,fontsize=16)
    ax.set_yticklabels(class_names,fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel('True Label',fontsize=20)
    ax.set_ylabel('Prediction',fontsize=20)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, str(mat[i, j]),
                    ha="center", va="center", fontsize=16,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    fig.tight_layout()
    plt.show()


# Compute moving average
def moving_average(a, n=3) :
    
    idx = np.cumsum(np.arange(len(a)),dtype=float)
    idx[n:] = idx[n:] - idx[:-n]
    
    res = np.cumsum(a, dtype=float)
    res[n:] = res[n:] - res[:-n]
    
    return idx[n - 1:] / n, res[n - 1:] / n


# Decorative progress bar
def progress_bar(count, total, message=''):
    """
    Args: count .... int/float, current progress counter
          total .... int/float, total counter
          message .. string, appended after the progress bar
    """
    from IPython.display import HTML, display,clear_output
    return HTML("""
        <progress 
            value='{count}'
            max='{total}',
            style='width: 30%'
        >
            {count}
        </progress> {frac}% {message}
    """.format(count=count,total=total,frac=int(float(count)/float(total)*100.),message=message))
    

def forward(blob,train=True):
    """
       Args: blob should have attributes, net, criterion, softmax, data, label
       
       Returns: a dictionary of predicted labels, softmax, loss, and accuracy
    """
    with torch.set_grad_enabled(train):
        # Prediction
        prediction = blob.net(blob.data)
        # Weight map
        classes,counts=torch.unique(blob.label,return_counts=True)
        weights=torch.ones(size=blob.label.size(),dtype=blob.data.dtype,device=blob.device)
        for i,c in enumerate(classes):
            weights[blob.label==c]=1./(len(classes)*counts[i].to(torch.float32))
        # Training
        loss,acc=-1,-1
        if blob.label is not None:
            loss = blob.criterion(prediction,blob.label)
        blob.loss = (loss * weights).sum()
        
        softmax    = blob.softmax(prediction).detach()
        prediction = torch.argmax(prediction,dim=1)
        accuracy   = ((prediction == blob.label)*weights).sum().item()
        
        
        return {'prediction' : prediction.detach(),
                'softmax'    : softmax,
                'loss'       : blob.loss.item(),
                'accuracy'   : accuracy}

def backward(blob):
    blob.optimizer.zero_grad()  # Reset gradients accumulation
    blob.loss.backward()
    blob.optimizer.step()
    

def train_loop(blob,train_loader,num_iteration):
    # Set the network to training mode
    blob.net.train()
    # Let's record the loss at each iteration and return
    train_loss=[]
    # Let's record the accuracy at each iteration and return
    train_acc=[]
    # Progress bar decoration
    from IPython.display import display
    progress=display(progress_bar(0,num_iteration),display_id=True)
    # Loop over data samples and into the network forward function
    clock_iter=time.time()
    time_record=[[],[],[],[]] # iteration, data read, cpu-gpu xfer, and computation
    while blob.iteration < num_iteration:
        for i,data in enumerate(train_loader):
            if blob.iteration >= num_iteration:
                break
            blob.iteration += 1
            # data and label
            if isinstance(data,dict):
                blob.data, blob.label = data['data'], data['label']
            else:
                blob.data, blob.label = data
            
            time_record[1].append(time.time() - clock_iter)

            clock_xfer = time.time()
            blob.data  = blob.data.to(blob.device)
            blob.label = blob.label.to(blob.device)
            time_record[2].append(time.time() - clock_xfer)
            
            # call forward
            clock_comp=time.time()
            res = forward(blob,True)
            # Record loss, accuracy
            train_loss.append(res['loss'])
            train_acc.append(res['accuracy'])
            # Backprop
            backward(blob)
            time_record[3].append(time.time() - clock_comp)
            
            time_record[0].append(time.time() - clock_iter)
            if blob.iteration%10 == 0:
                # once in a while, report
                message='Iteration: %d elapsed %d [sec] ... Loss: %.2f Accuracy: %.2f'
                message = message % (blob.iteration,
                                     int(np.sum(time_record[0])),
                                     res['loss'],
                                     res['accuracy'])
                progress.update(progress_bar(blob.iteration,num_iteration,message=message))
                
            clock_iter=time.time()
            
    progress.update(progress_bar(num_iteration,num_iteration,message=message))
    return dict(loss=np.array(train_loss),
                accuracy=np.array(train_acc),
                time_iteration=np.array(time_record[0]),
                time_data_read=np.array(time_record[1]),
                time_data_xfer=np.array(time_record[2]),
                time_compute  =np.array(time_record[3]))


def plot_loss(loss,num_average=30,iterations_per_epoch=None):
    import numpy as np
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,8),facecolor='w')
    iterations=np.array(range(len(loss))).astype(np.float32)
    if iterations_per_epoch is not None:
        iterations = iterations / iterations_per_epoch
    
    plt.plot(iterations,loss,marker="",linewidth=2,color='blue',label='loss (raw)')
    plt.plot(moving_average(iterations,num_average),moving_average(loss,num_average),
             marker="",linewidth=2,color='red',label='rolling mean')
    ax.set_xlabel('Iterations' if iterations_per_epoch is None else 'Epoch',fontsize=20)
    ax.set_ylabel('Loss',fontsize=20)
    plt.tick_params(labelsize=20)
    plt.grid(True,which='both')
    leg=plt.legend(fontsize=16,loc=2)
    leg_frame=leg.get_frame()
    leg_frame.set_facecolor('white')
    plt.show()


def inference_loop(blob,dataloader,local_data_dir='./',num_iterations=100):
    import numpy as np
    # set the network to test (non-train) mode
    blob.net.eval()
    # create the result holder
    accuracy, label, prediction, softmax = [], [], [], []
    confusion_matrix = np.zeros([10,10],dtype=np.int32)
    for i,data in enumerate(dataloader):
        if isinstance(data,dict):
            blob.data, blob.label = data['data'], data['label']
        else:
            blob.data, blob.label = data
            
        blob.data  = blob.data.to(blob.device)
        blob.label = blob.label.to(blob.device)
        
        res = forward(blob,False)
        accuracy.append(res['accuracy'])
        prediction.append(res['prediction'].detach())
        label.append(blob.label.detach())
        softmax.append(res['softmax'].detach())
        if i >= num_iterations:
            break
    # organize the return values
    accuracy   = np.hstack(accuracy)
    prediction = torch.cat(prediction,dim=0).cpu().numpy()
    label      = torch.cat(label,dim=0).cpu().numpy()
    softmax    = torch.cat(softmax,dim=0).cpu().numpy()
    return accuracy, label, prediction, softmax
