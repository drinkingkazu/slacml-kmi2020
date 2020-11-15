import numpy as np

def plot_dataset(dataset,num_image_per_class=10):
    import numpy as np
    num_class = 0
    classes = []
    if hasattr(dataset,'classes'):
        classes=dataset.classes
        num_class=len(classes)
    else: #brute force
        for data,label in dataset:
            if label in classes: continue
            classes.append(label)
        num_class=len(classes)
    
    shape = dataset[0][0].shape
    big_image = np.zeros(shape=[3,shape[1]*num_class,shape[2]*num_image_per_class],dtype=np.float32)
    
    finish_count_per_class=[0]*num_class
    for data,label in dataset:
        if finish_count_per_class[label] >= num_image_per_class: continue
        img_ctr = finish_count_per_class[label]
        big_image[:,shape[1]*label:shape[1]*(label+1),shape[2]*img_ctr:shape[2]*(img_ctr+1)]=data
        finish_count_per_class[label] += 1
        if np.sum(finish_count_per_class) == num_class*num_image_per_class: break
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,8),facecolor='w')
    ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelleft=False,labelbottom=False)
    plt.imshow(np.transpose(big_image,(1,2,0)))
    for c in range(len(classes)):
        plt.text(big_image.shape[1]+shape[1]*0.5,shape[2]*(c+0.6),str(classes[c]),fontsize=16)
    plt.grid(False)
    plt.show()

class ImagePadder(object):
    
    def __init__(self,padsize=20,randomize=False):
        self._pads=padsize
        self._randomize=randomize
        
    def __call__(self,img):
        import numpy as np
        img=np.array(img)
        new_shape = list(img.shape)
        new_shape[0] += self._pads
        new_shape[1] += self._pads
        
        new_img = np.zeros(shape=new_shape,dtype=img.dtype)
        pad_vertical   = int(self._pads / 2.)
        pad_horizontal = int(self._pads / 2.)
        if self._randomize:
            pad_vertical   = int(np.random.uniform() * self._pads)
            pad_horizontal = int(np.random.uniform() * self._pads)
        
        new_img[pad_vertical:pad_vertical+img.shape[0],pad_horizontal:pad_horizontal+img.shape[1]] = img
        return new_img
