import torch

class MyModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(in_features=20,out_features=10)
        self.layer1 = torch.nn.Linear(in_features=10,out_features=5 )
        self.layer2 = torch.nn.Linear(in_features=5, out_features=1 )
        
    def forward(self,data):
        
        print('\nInput  shape', data.shape)
        
        data = self.layer0(data)
        print('\nAfter layer0', data.shape)
        
        data = self.layer1(data)
        print('\nAfter layer1', data.shape)
        
        data = self.layer2(data)
        print('\nAfter layer2', data.shape)
        
        return data


if __name__ == '__main__':
    
    data = torch.randn(10,20)
    
    model = MyModel()
    model(data)