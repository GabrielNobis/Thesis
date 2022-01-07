import numpy as np
import torch

from abc import ABC, abstractmethod
from triplet import bm_triplet

class Minibatch(ABC):
    
    """
    Simulates a D-dimensionl Ito-process on [0,T] over N time steps
    """
    
    def __init__(self, T=1, N=100, D=1,s0=0):
        
        self.T = T  #terminal time
        self.N = N  #number of time steps
        self.D = D  #dimension of the process 
    
        self.dt = T/N 
        
        self.s0 = s0
        
    def fetch(self,M):
        
        Dt = torch.zeros((M,self.N+1,self.D)) # M x (N+1) x D; m-the row corresponds to time steps used for m-th trajectory
        DX = torch.zeros((M,self.N+1,self.D)) # M x (N+1) x D; m-th row corresponds to m-th trajectory         
        
        Dt[:,1:,:] = self.dt
        DX[:,1:,:] = self.increment(M)

        t = torch.cumsum(Dt,axis=1) # M x (N+1) x 1; time grid
        X = torch.cumsum(DX,axis=1) # M x (N+1) x D; D-dim. realisation of trajectory of BM  
        
        return t, X, DX
    
    @abstractmethod
    def increment(self, M):
        return torch.zeros((M,self.N,self.D))
                
        #specify the distribution of the increment
     #   dX = torch.normal(torch.zeros((M,self.N,self.D)),torch.ones((M,self.N,self.D)))
             
     #   return np.sqrt(self.dt)*dX
    
    @abstractmethod
    def path(self,M):    
        
        S = torch.ones((M,self.N+1,self.D))*self.s0
        t,X,dX = self.fetch(M)
        
        t0 = t[:,0,:]
        X0 = X[:,0,:]
        
        for n in range(self.N):
            
            t1 = t[:,n+1,:]
            X1 = X[:,n+1,:]

            S[:,n+1,:] = self.step(t0,S[:,n,:],(t1-t0),(X1-X0)) 
            
            t0=t1
            X0=X1
        
        return S
    
    @abstractmethod
    def step(self,t,s,h,dX):
        return s + self.drift(t,s)*h + self.vol(t,s)*dX
    
    @abstractmethod
    def drift(self,t,S):
        return torch.zeros_like(S)
    
    @abstractmethod
    def vol(self,t,S):
        return torch.ones_like(S)

class DetProcess(Minibatch):
    
    def __init__(self, T=1, N=100, D=1,s0=0):
        super().__init__(T=T, N=N, D=D,s0=s0)
        
    def increment(self,M):
        return super().increment(M)
    
    def path(self,M):
        return super().path(M)
    
    def step(self,t,x,h,dX):
        return super().step(t,x,h,dX)
    
    def drift(self,t,S):
        return 2*S+1
    
    def vol(self,t,S):
        return super().vol(t,S)

class Bm(Minibatch):
    
    def __init__(self, T=1, N=100, D=1, w0=0):
        super().__init__(T=T, N=N, D=D, s0=w0)
        
    def increment(self,M):
        dX = torch.normal(torch.zeros((M,self.N,self.D)),torch.ones((M,self.N,self.D)))
        return np.sqrt(self.dt)*dX
    
    def path(self,M):
        return super().path(M)
    
    def step(self,t,x,h,dX):
        return super().step(t,x,h,dX)
    
    def drift(self,t,S):
        return super().drift(t,S)
    
    def vol(self,t,S):
        return super().vol(t,S)

class gBm(Bm):
    
    def __init__(self, T=1, N=100, D=1, p0=1,mu=0.08,sigma=0.4):
        super().__init__(T=T, N=N, D=D, w0=p0)
        
        self.mu = mu
        self.sigma = sigma
    
    def drift(self,t,s):
        return self.mu*s
    
    def vol(self,t,s):
        return self.sigma*s

class gBmTriplet(gBm):
    
    def __init__(self, T=1, N=100, D=1,p0=1,mu=0.08,sigma=0.4):
        super().__init__(T=T, N=N, D=D,p0=p0,mu=mu,sigma=sigma)
    
    def full_path(self,M):
        
        S = torch.ones((M,self.N+1,self.D))*self.s0
        P = torch.ones((M,self.N+1,self.D))*self.s0
        W = torch.zeros((M,self.N+1,self.D))*self.s0
        
        rm = torch.zeros((M,self.N+1,self.D))
        rM = torch.zeros((M,self.N+1,self.D))
        
        erm = torch.ones((M,self.N+1,self.D))
        erM = torch.ones((M,self.N+1,self.D))
        
        t,X,dX = self.fetch(M)
        
        t0 = t[:,0,:]
        X0 = X[:,0,:]
        
        for n in range(self.N):
            
            t1 = t[:,n+1,:]
            X1 = X[:,n+1,:]
            
            dX = X1-X0
            
            P[:,n+1,:] = self.step(t0,P[:,n,:],(t1-t0),dX) 
            W[:,n+1,:],rm[:,n+1,:],rM[:,n+1,:] = bm_triplet(torch.log(S[:,n,:]),d=(self.mu-((self.sigma**2)/2)),v=self.sigma,X=(dX/np.sqrt(self.dt)),T=self.dt)
            S[:,n+1,:] =  torch.exp(W[:,n+1,:])#+W[:,n+1,:])
            
            erm[:,n+1,:] = torch.exp(rm[:,n+1,:])
            erM[:,n+1,:] = torch.exp(rM[:,n+1,:])
            
            t0=t1
            X0=X1
        
        return W, rm, rM, S, erm, erM, P
