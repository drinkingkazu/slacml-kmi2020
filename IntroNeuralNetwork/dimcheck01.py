import torch

class MyModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.pool=torch.nn.MaxPool2d(2,2)
        self.layer0=torch.nn.Conv2d(in_channels=1, out_channels=16,kernel_size=3,padding=1)
        self.layer1=torch.nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1)
        self.layer2=torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        self.layer3=torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1)
        
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3]
        
    def forward(self,data):
        
        print('\nInput  shape', data.shape)
        
        for i in range(len(self.layers)):
            data = self.layers[i](data)
            print('\nAfter layer',i,'shape',data.shape)
            data = self.pool(data)
            print('... and pool2d','shape',data.shape)
        
        return data


if __name__ == '__main__':
    
    data = torch.randn(2*32*32).reshape(2,1,32,32)
    
    model = MyModel()
    model(data)