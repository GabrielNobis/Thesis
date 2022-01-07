import torch
import torch.nn as nn


# +
class ReluDrift(nn.Module):
    
    def __init__(self,d=3,n=8):

        super(ReluDrift,self).__init__()
        
        self.fc1 = nn.Linear(d,n)
        self.fc2 = nn.Linear(n,n)
        self.fc3 = nn.Linear(n,n)
        self.fc4 = nn.Linear(n,n)
        self.fc5 = nn.Linear(n,1)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        
        x = self.fc5(x)
        
        return x

class TanhDrift(nn.Module):
    
    def __init__(self,d=3,n=8):

        super(TanhDrift,self).__init__()
        
        self.fc1 = nn.Linear(d,n)
        self.fc2 = nn.Linear(n,n)
        self.fc3 = nn.Linear(n,n)
        self.fc4 = nn.Linear(n,n)
        self.fc5 = nn.Linear(n,1)
        
    def forward(self,x):
        
        x = self.fc1(x)
        x = torch.tanh(x)
        
        x = self.fc2(x)
        x = torch.tanh(x)
        
        x = self.fc3(x)
        x = torch.tanh(x)
        
        x = self.fc4(x)
        x = torch.tanh(x)
        
        x = self.fc5(x)
        
        return x

# +
class ReluDiffusion(nn.Module):
    
    def __init__(self,d=3,n=8):

        super(ReluDiffusion,self).__init__()
        self.fc1 = nn.Linear(d,n)
        self.fc2 = nn.Linear(n,n)
        self.fc3 = nn.Linear(n,n)
        self.fc4 = nn.Linear(n,n)
        self.fc5 = nn.Linear(n,1)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
    
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x
    
class SoftSigmoidDiffusion(nn.Module):
    
    def __init__(self,d=3,n=8):

        super(SoftSigmoidDiffusion,self).__init__()
        self.fc1 = nn.Linear(d,n)
        self.fc2 = nn.Linear(n,n)
        self.fc3 = nn.Linear(n,n)
        self.fc4 = nn.Linear(n,n)
        self.fc5 = nn.Linear(n,1)
        self.Softplus = torch.nn.Softplus()

    def forward(self,x):
    
        x = self.fc1(x)
        x = self.Softplus(x)
        x = self.fc2(x)
        x = self.Softplus(x)
        x = self.fc3(x)
        x = self.Softplus(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        x = self.fc5(x)
        return x
