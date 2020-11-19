import numpy as np
import torch
import torch.nn.functional as F

class ResNetModule(torch.nn.Module):

    def __init__(self, num_input, num_output, stride=1, momentum=0.9 ):
        super(ResNetModule, self).__init__()

        # residual path
        self._features = torch.nn.Sequential(
            torch.nn.Conv2d(num_input, num_output, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_output,momentum=momentum),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_output, num_output, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_output,momentum=momentum)
        )

        # if stride >1, then we need to subsamble the input
        if stride>1 or not num_input==num_output:
            self._shortcut = torch.nn.Conv2d(num_input,num_output,kernel_size=1,stride=stride,bias=False)
        else:
            self._shortcut = None
            

    def forward(self, x):

        if self._shortcut is None:
            bypass = x
        else:
            bypass = self._shortcut(x)

        residual = self._features(x)

        return torch.nn.ReLU(inplace=True)(bypass + residual)

class ResNetLayer(torch.nn.Module):

    def __init__(self, num_input, num_output, num_modules, stride=1, momentum=0.9):

        super(ResNetLayer,self).__init__()

        ops = [ ResNetModule(num_input, num_output, stride=stride, momentum=momentum) ]

        for i in range(num_modules-1):

            ops.append( ResNetModule(num_output, num_output, stride=1, momentum=momentum) )

        self._layer = torch.nn.Sequential(*ops)

    def forward(self,x):
        return self._layer(x)

class ResNet(torch.nn.Module):

    def __init__(self, num_class, num_input, num_output_base, blocks, bn_momentum=0.9):
        """
        Args: num_class ... integer, number of filters in the last layer
              num_input ... integer, number of channels in the input data
              num_output_base ... integer, number of filters in the first layer
              blocks ... list of integers, number of ResNet modules at each spatial dimensions
         """

        super(ResNet, self).__init__()
        
        self._ops = []
        
        num_output = num_output_base

        for block_index, num_modules in enumerate(blocks):

            stride = 2 if block_index > 0 else 1

            self._ops.append( ResNetLayer(num_input, num_output, num_modules, stride=stride, momentum=bn_momentum) )
            
            # For the next layer, increase channel count by 2
            num_input  = num_output
            num_output = num_output * 2
            
        self._features = torch.nn.Sequential(*self._ops)

        self._classifier = torch.nn.Linear(num_input, num_class)

    def forward(self,x,show_shape=False):
        
        tensor = x
        if not show_shape:
            tensor = self._features(tensor)
        else:
            print('Input shape', tensor.size())
            for block_ctr, op in enumerate(self._ops):
                tensor = op(tensor)
                print('After block',block_ctr,tensor.size())
        
        tensor = F.max_pool2d(tensor, kernel_size=tensor.size()[2:])
        if show_shape: print('After average pooling',tensor.size())
        
        tensor = tensor.view(-1,np.prod(tensor.size()[1:]))
        if show_shape: print('After reshape',tensor.size())
        
        tensor = self._classifier(tensor)
        if show_shape: print('After classifier',tensor.size())
        
        return tensor

if __name__ == '__main__':

    num_class  = 3
    num_input  = 1
    num_output_base = 16
    num_modules = [2,2,2,2,2]

    tensor = torch.Tensor(np.zeros(shape=[10,1,192,192]))
    net = ResNet(num_class, num_input, num_output_base, num_modules)

    net(tensor,show_shape=True)