import numpy as np
from resnet import ResNetLayer 
import torch
import torch.nn.functional as F

class UResNet(torch.nn.Module):

    def __init__(self, num_class, num_input, num_output_base, blocks, bn_momentum=0.9):
        """
        Args: num_class ... integer, number of filters in the last layer
              num_input ... integer, number of channels in the input data
              num_output_base ... integer, number of filters in the first layer
              blocks ... list of integers, number of ResNet modules at each spatial dimensions
         """

        super(UResNet, self).__init__()
        
        self._encoder  = []
        self._upsampler= []
        self._decoder  = []
        
        num_output = num_output_base
        features = []
        for block_index, num_modules in enumerate(blocks):

            stride = 2 if block_index > 0 else 1

            self._encoder.append( ResNetLayer(num_input, 
                                              num_output, 
                                              num_modules, 
                                              stride=stride, 
                                              momentum=bn_momentum) 
                                )           
            # For the next layer, increase channel count by 2
            features.append((num_input,num_output))
            num_input  = num_output
            num_output = num_output * 2
            
        for i in range(len(features)-1):
            num_output,num_input = features[-1*(i+1)]
            num_modules = blocks[-1*i]
            self._upsampler.append(torch.nn.ConvTranspose2d(num_input,
                                                            num_output,
                                                            3, 
                                                            stride=2, 
                                                            padding=1)
                                  )
            self._decoder.append( ResNetLayer(num_output*2,
                                              num_output,
                                              num_modules,
                                              stride=1,
                                              momentum=bn_momentum))
            
        ops=[]
        for op in self._encoder: ops.append(op)
        for op in self._decoder: ops.append(op)
        for op in self._upsampler: ops.append(op)
        self._ops = torch.nn.Sequential(*ops)
        
        self._classifier = torch.nn.Conv2d(num_output_base,
                                           num_class, 
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False)

    def forward(self,x,show_shape=False):
        
        features = [x]
        if show_shape: print('Input ...',x.shape)
        for i,module in enumerate(self._encoder):
            features.append(module(features[-1]))
            if show_shape: print('After encoder block',i,'...',features[-1].shape)
            
        decoder_input = features[-1]
        
        for i,module in enumerate(self._decoder):
            decoder_input = self._upsampler[i](decoder_input, output_size=features[-1*(i+2)].size())
            if show_shape: print('After upsample',i,'...',decoder_input.shape)
            decoder_input = torch.cat([decoder_input,features[-1*(i+2)]],dim=1)
            if show_shape: print('After concat  ',i,'...',decoder_input.shape)
            decoder_input = self._decoder[i](decoder_input)
            if show_shape: print('After decoder ',i,'...',decoder_input.shape)
            
        result = self._classifier(decoder_input)
        if show_shape: print('Result',result.shape)
        
        return result

if __name__ == '__main__':

    num_class  = 3
    num_input  = 1
    num_output_base = 16
    num_modules = [2,2,2,2,2]

    tensor = torch.Tensor(np.zeros(shape=[10,1,256,256])).cuda()
    net = UResNet(num_class, num_input, num_output_base, num_modules).cuda()
    net(tensor,show_shape=True)