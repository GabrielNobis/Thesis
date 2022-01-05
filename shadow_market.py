import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

import torch
import torch.nn as nn
import torch.optim as optim
import time
import importlib

import coefficient
importlib.reload(coefficient)
from coefficient import *

import utils
importlib.reload(utils)
from utils import approx_m, plot_learning_summary

from triplet import bm_triplet



SellBoundary = np.flip(np.array(sio.loadmat('SellBoundary_100.mat')['SellBoundary']).T)
BuyBoundary = np.flip(np.array(sio.loadmat('BuyBoundary_100.mat')['BuyBoundary']).T)

class Alpha():
    
    def __init__(self,smooth_f=False,smooth_df=False,smooth_ddf=False):
        
        self.f = torch.nn.Hardtanh()
        self.df = self.Dhardtah
        self.ddf = lambda x: torch.zeros_like(x)
        
        if smooth_f:
            self.f = torch.tanh
            
        if smooth_df:
            self.df = lambda x: (1-torch.tanh(x)**2)
            
        if smooth_ddf:
            self.ddf = lambda x:(-2*torch.tanh(x))*(1-torch.tanh(x)**2)
    
    def Dhardtah(self,x):
        
        thr1 = torch.nn.Threshold(1, 1)
        thr2 = torch.nn.Threshold(-1.0000001, 0)
        z = thr1(torch.abs(x))
        return -g(-z)

class Shadow_Price():
    
    def __init__(self, T=1, N=100, D=1, M=1000, P0=1, mu=0.08, sigma=0.4, gamma=0.01, x=1,learn_s=True,s=1.021, smooth_f=False,smooth_df=True,smooth_ddf=False,method='baseline', hidden_formular='explicit',nn_in_dim = 2,activation='relu'):
        
        self.T = T # terminal time
        self.N = N # number of time snapshots
        self.dt = T/N
        self.D = D # number of dimensions
        self.M = M # number of trajectories
        #model parameters
        self.p0 = P0 
            
        self.mu = mu
        self.mu_hat = mu
        
        self.sigma = sigma
        self.sigma_hat = sigma
        
        self.gamma = gamma
        self.x = x
        
        self.learn_s = learn_s
        self.s = torch.tensor(s,dtype=torch.float,requires_grad=self.learn_s)

        self.nn_in_dim = nn_in_dim
        
        if activation == 'non_relu':
            self.drift = TanhDrift(d=nn_in_dim)
            self.diffusion = SoftSigmoidDiffusion(d=nn_in_dim)
        else:
            self.drift = ReluDrift(d=nn_in_dim)
            self.diffusion = ReluDiffusion(d=nn_in_dim)

        self.stamp = str(time.time() - int(time.time()))[2:7]
        
        self.training_loss = []
        self.used_lr = []
        self.used_iter = []
        
        self.intermediate_pi = []    
        
        self.smooth_ddf = smooth_ddf
        self.transformation = Alpha(smooth_f=smooth_f,smooth_df=smooth_df,smooth_ddf=smooth_ddf)
        
        self.g = lambda p,x: p + self.gamma*p*self.transformation.f(x)
        
        self.method = method
        self.hidden_formular = hidden_formular
    
    def stanh(self,X):
        
        c=1/0.9640
        htanh = torch.nn.Hardtanh()
        X = 2*htanh(0.5*X)
        X = c*torch.tanh(X)
        return X
    
    def weights_init_normal(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.normal_(m.weight,std=np.sqrt(self.gamma))
            torch.nn.init.normal_(m.bias,std=np.sqrt(self.gamma))

    def weights_init(self):
        self.drift.apply(self.weights_init_normal)
        self.diffusion.apply(self.weights_init_normal)
        return
        
    def net(self,t,Sp,m,S,X):
        
        Y = Sp/m
        
        if self.nn_in_dim==2:
            input_d = torch.cat([t,Y],1)
            d = self.drift(input_d)
        
            input_v = torch.cat([t,Y],1)
            v = self.diffusion(input_v)
            
        if self.nn_in_dim==3:
            input_d = torch.cat([t,Y,m],1)
            d = self.drift(input_d)
        
            input_v = torch.cat([t,Y,m],1)
            v = self.diffusion(input_v)
            
        if self.nn_in_dim==4:
            input_d = torch.cat([t,Y,m,Sp],1)
            d = self.drift(input_d)
        
            input_v = torch.cat([t,Y,m,Sp],1)
            v = self.diffusion(input_v)
            
        return d,v
    
    def hiddenX(self,t,P,S,X,m,m1,M1):
        
        Sp = (1+self.gamma)*P
        m  = approx_m(self.s, Sp,m,torch.exp(m1),torch.exp(M1))
        
        d,v = self.net(t,Sp,m[0],S,X)
        
        if self.hidden_formular == 'explicit':
            
            Xd = (S/(P))*(self.sigma**2 - self.mu) + d #BABSIS
            Xv = (S/(P))*(-self.sigma)  + v #BABSIS
        
        elif self.hidden_formular =='unexplicit':
            Xd = d
            Xv = v
            
        return Xd,Xv,m,d,v
    
    def coeff(self, d, v, X, P, S,m):
        
        f = self.transformation.f
        df = self.transformation.df
        ddf = self.transformation.ddf
        
        if self.method != 'oracle':
            
            a = self.mu + (df(X)/(1+self.gamma*f(X)))*d + (df(X)/(1+self.gamma*f(X)))*self.sigma*v + self.smooth_ddf*(1/(2*self.gamma))*(ddf(X)/(1+self.gamma*f(X)))*(v**2)
            b = self.sigma + (df(X)/(1+self.gamma*f(X)))*v
            
        if self.method == 'oracle':
            
            self.dxg = lambda x,y: self.gamma*y*df(x)
            self.dydxg = lambda x,y: self.gamma*df(x)
            self.dxdxg = lambda x,y: self.gamma*y*ddf(x)
        
            self.dyg = lambda x,y: 1 + self.gamma*f(x)
            self.dxdyg = lambda x,y: + self.gamma*df(x)
            self.dydyg = lambda x,y: torch.zeros_like(x)
            
            a = (self.dxg(X,P)/S)*d + (self.dyg(X,P)/S)*P*self.mu + ((self.dydxg(X,P)/S)*P*self.sigma*v) + 0.5*(self.dxdxg(X,P)/S)*(v**2)
            b = (self.dxg(X,P)/S)*v + (self.dyg(X,P)/S)*P*self.sigma
            
          #  a = P*((self.dxg(X,P)/S)*d + (self.dyg(X,P)/S)*self.mu + ((self.dydxg(X,P)/S)*self.sigma*v) + 0.5*(self.dxdxg(X,P)/S)*P*(v**2)) #correct adjustment
          #  b = P*((self.dxg(X,P)/S)*v + (self.dyg(X,P)/S)*self.sigma) #correct adjustment

        return a,b

    def theta_and_pi(self,a,b):
        return a/b, a/(b**2)

    def strategy(self,pi,X,S):
        with torch.no_grad():
            sr = pi*X/S
            sb = X - sr*S
        return sb, sr
    
    def transform_etw(self, etw):
        return np.log(self.x/(self.T+1)) + 0.5*etw

    def fraction_P(self, pi, S, P):
        rw = (pi/(1-pi))*(P/S)
        return rw/(1+rw)
    
    def hidden_step(self,h,dW,X0,Xd,Xv,S0,P0):
        
        if self.method == 'baseline':
            X1 = X0 + (1/(self.gamma))*(Xd*h + Xv*dW)
        
        if self.method == 'oracle':
            X1 = X0 + P0*(Xd*h + Xv*dW)
        
        return X1
    def loss_path(self, t, W):
        
        N=self.N
        h = self.dt
        t0 = t[:,0,:]
        W0 = W[:,0,:]
        
        P0 = torch.ones(self.M,self.D)*self.p0
        m0 = torch.zeros(self.M,self.D)
        M0 = torch.zeros(self.M,self.D)
        m = ((1+self.gamma)*torch.exp(m0),torch.ones((self.M,1), dtype=torch.bool),torch.ones((self.M,1), dtype=torch.bool))
        X0 = (2*torch.rand(size=(self.M,self.D)) -1)*0 
        S0 = self.g(P0,X0)
        
        Xd,Xv,m,_,_ = self.hiddenX(t0,P0,S0,X0,m,m0,M0)
        a,b = self.coeff(Xd,Xv,X0,P0,S0,m)
        
        loss_expectation = 0
        for n in range(N):
        
            t1 = t[:,n+1,:]
            W1 = W[:,n+1,:]
            
            dW = (W1-W0) 
            
            theta,_ = self.theta_and_pi(a,b)
            loss_expectation += torch.mean(theta**2)*h 
    
            Wx,m1,M1 = bm_triplet(np.log((1+self.gamma)*P0),d=(self.mu-((self.sigma**2)/2)),v=self.sigma,X=(dW/np.sqrt(h)),T=h)
            P1 = (torch.exp(Wx)/(1+self.gamma))
            
            X1 = self.hidden_step(h,dW,X0,Xd,Xv,S0,P0)
            S1 = self.g(P1,X1)
         
            Xd,Xv,m,_,_ = self.hiddenX(t1,P1,S1,X1,m,m1,M1)
            a,b = self.coeff(Xd,Xv,X1,P1,S1,m)
            
            P0 = P1
            X0 = X1
            S0 = S1
            t0 = t1
            W0 = W1
            
            m0 = m1
            M0 = M1
        
        return loss_expectation + self.learn_s*torch.relu(-self.s+1.021)
    
    def fetch_minibatch(self, M):
        
        '''
        From Raissi 2018
        
        :return:
            t: M x (N+1) x 1 array; m-the row corresponds to time steps used for m-th trajectory
            W: M x (N+1) x D array; m-th row corresponds to a D-dim. realisation of trajectory of BM
        '''
        T = self.T
        D = self.D #number of dimensions
        N = self.N

        Dt = np.zeros((M,N+1,1)) # M x (N+1) x 1; m-the row corresponds to time steps used for m-th trajectory
        DW = np.zeros((M,N+1,D)) # M x (N+1) x D; m-th row corresponds to m-th trajectory
        
        Dt[:,1:,:] = self.dt
        DW[:,1:,:] = np.sqrt(self.dt)*np.random.normal(size=(M,N,D))  #~N(0,dt*Id)

        t = np.cumsum(Dt,axis=1) # M x (N+1) x 1; time grid
        W = np.cumsum(DW,axis=1) # M x (N+1) x D; D-dim. realisation of trajectory of BM
        
        t = torch.tensor(t,dtype=torch.float)
        W = torch.tensor(W,dtype=torch.float)
        
        return t, W

    def train(self,Iter,lrs=[1e-3],plot=False,show_plot=100,plot_loss=False,interstep=30*10**3):
        
        self.used_lr += lrs
        self.used_iter.append(Iter)
        
        for i in range(len(lrs)):
            
            params = list(self.drift.parameters()) + list(self.diffusion.parameters()) + self.learn_s*[self.s]
            optimizer = optim.Adam(params,lr=lrs[i])
                  
            for it in range(Iter):
                running_loss = 0.0
                
                optimizer.zero_grad()

                t,W = self.fetch_minibatch(self.M)
                loss = self.loss_path(t,W)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                self.training_loss.append(loss.item())  
                
                if it%10==0:
                    print('loss=%.4e - loss=%.4e - Iter %i with lr=%.3e'%(running_loss/10,loss/10,it,lrs[i]))
                    print('upper bound for Y:',self.s)
                    running_loss = 0.0

                if it%show_plot==0 and plot:
                    self.plot_processes(100)
                    self.plot_training_loss()
                    
                if it%interstep==0:
                    X, S, P, ETW, PI, PI_t, SB, SR, LOSS, A, B, Y = self.predict(100)
                    self.intermediate_pi.append(PI_t)
                
        if plot_loss:
            self.plot_training_loss()
            
        return
    
    def predict(self, M):
        
        N = self.N
        h = self.dt
        
        with torch.no_grad():
            
            t, W = self.fetch_minibatch(M)
            t0 = t[:,0,:]
            W0 = W[:,0,:]
            
            
            P = torch.ones(M,N+1,1)*self.p0 #price in market with frictions
            X = torch.zeros(M,N+1,1) #hidden process
            S = torch.ones(M,N+1,1)*self.p0 #shadow price
            Y = torch.ones(M,N+1,1)
            V = torch.ones(M,N+1,1) #wealth process
            
            
            V[:,0,:] = torch.ones_like(P[:,0,:])*self.x
            X[:,0,:] = (2*torch.rand(size=(M,self.D)) -1)*0
            S[:,0,:] = self.g(P[:,0,:],X[:,0,:])
            
            m0 = torch.zeros(M,self.D)
            M0 = torch.zeros(M,self.D)
            m = ((1+self.gamma)*torch.exp(m0),torch.ones((M,1), dtype=torch.bool),torch.ones((M,1), dtype=torch.bool))
            
            PI = torch.zeros(M,N+1,1)
            SB = torch.zeros(M,N+1,1)
            SR = torch.zeros(M,N+1,1)
            
            self.drift_val = torch.zeros(M,N+1,1)
            self.diffusion_val = torch.zeros(M,N+1,1)
            
            A = torch.ones(M,N+1,1)
            B = torch.ones(M,N+1,1)
            
            SB = torch.zeros(M,N+1,1)
            SR = torch.zeros(M,N+1,1)
            
            LOSS = torch.zeros(N)
            
                    
            Xd,Xv,m,self.drift_val[:,0,:],self.diffusion_val[:,0,:] = self.hiddenX(t0,P[:,0,:],S[:,0,:],X[:,0,:],m,m0,M0)
            A[:,0,:], B[:,0,:] = self.coeff(Xd,Xv,X[:,0,:],P[:,0,:],S[:,0,:],m)
            
            PI[:,0,:] = torch.ones(M,1)*(self.mu/(self.sigma**2))
            SB[:,0,:], SR[:,0,:] = self.strategy(PI[:,0,:], V[:,0,:], S[:,0,:])
            
            self.PI_t = torch.zeros_like(PI)
            self.PI_t[:,0,:] = PI[:,0,:]
            
            PI_t = self.PI_t
            
            ETW = 0
            for n in range(self.N):
            
                t1 = t[:,n+1,:]
                W1 = W[:,n+1,:]

                dW = W1 - W0
                
                theta, PI[:,n,:] = self.theta_and_pi(A[:,n,:],B[:,n,:])
                PI_t[:,n,:] = self.fraction_P(PI[:,n,:],S[:,n,:],(1+self.gamma)*P[:,n,:])
                SB[:,n,:], SR[:,n,:] = self.strategy(PI[:,n,:], V[:,n,:], S[:,n,:])
                V[:,n+1,:] = V[:,n,:] + (V[:,n,:]*PI[:,n,:])*(A[:,n,:]*h + B[:,n,:]*dW)
                LOSS[n] =  torch.mean(theta**2)*h
                ETW += LOSS[n]
                
                Wx,m1,M1 = bm_triplet(np.log((1+self.gamma)*P[:,n,:]),d=(self.mu-((self.sigma**2)/2)),v=self.sigma,X=(dW/np.sqrt(h)),T=h)
                P[:,n+1,:] = (torch.exp(Wx)/(1+self.gamma))
                
                X[:,n+1,:] = self.hidden_step(h,dW,X[:,n,:],Xd,Xv,S[:,n,:],P[:,n,:])
                S[:,n+1,:] = self.g(P[:,n+1,:],X[:,n+1,:])                
                
                Xd,Xv, m,self.drift_val[:,n+1,:],self.diffusion_val[:,n+1,:]= self.hiddenX(t1,P[:,n+1,:],S[:,n+1,:],X[:,n+1,:],m,m1,M1)
                A[:,n+1,:], B[:,n+1,:]= self.coeff(Xd,Xv,X[:,n+1,:],P[:,n+1,:],S[:,n+1,:],m)                
                Y[:,n+1,:] = ((1+self.gamma)*P[:,n+1,:])/m[0]
                
                t0 = t1
                W0 = W1    

            ETW = self.transform_etw(ETW)
            theta, PI[:,n+1,:] = self.theta_and_pi(A[:,n+1,:],B[:,n+1,:])
            PI_t[:,n+1,:] = self.fraction_P(PI[:,n+1,:],S[:,n+1,:],(1+self.gamma)*P[:,n+1,:])
            SB[:,n+1,:], SR[:,n+1,:] = self.strategy(PI[:,n+1,:], V[:,n+1,:], S[:,n+1,:])
        
        return X, S, P, ETW, PI, PI_t, SB, SR, LOSS, A, B, Y
    
    def plot_processes(self,M,save=False,heatmap=False,heat_path=None,name='shadow_price'):
        
        X, S, P, ETW, PI, PI_t, SB, SR, LOSS, A, B, Y = self.predict(M)
        
        x_axes = np.arange(self.N+1)/(self.N+1)
        fig = plt.figure(figsize=(25,25))
        fig.suptitle('%i simulations of learned processes - initial capital: %.1f - eotw: %.4e'%(M,self.x,ETW),fontsize=25)
        
        ax1 = fig.add_subplot(331)
        ax1.title.set_text(r'values of $\bar{X}$')
        ax1.set_ylabel(r'$\bar{S}_{t}$')
        ax1.set_xlabel(r'time t')
        
        ax2 = fig.add_subplot(332)
        ax2.title.set_text('optimal strategy')
        ax2.set_ylabel('strategy')
        ax2.set_xlabel(r'time t') 
        
        ax3 = fig.add_subplot(333)
        ax3.title.set_text(r'values of $\pi$ with transaction costs')
        ax3.set_ylabel(r'risky fraction $\pi$')
        ax3.set_xlabel(r'time t')
         
        
        ax4 = fig.add_subplot(334)
        ax4.title.set_text(r'values of $\bar{Y}$')
        ax4.set_xlabel(r'time t')
        
        ax5 = fig.add_subplot(335)
        ax5.title.set_text(r'values of $R_{1}(\Phi_{1})$')
        ax5.set_xlabel(r'time t')
        
        ax6 = fig.add_subplot(336)
        ax6.title.set_text(r'values of $R_{2}(\Phi_{1})$')
        ax6.set_xlabel(r'time t')
                           
        
        ax7 = fig.add_subplot(337)
        ax7.title.set_text(r'values of $\hat{\mu}$')
        ax7.set_xlabel(r'time t')
        '''
        ax6 = fig.add_subplot(4,3,11)
        ax6.title.set_text(r'$\pi^{\star}$ in shadow market')
        ax6.plot(SellBoundary,color='darkred',linewidth=2,label='Sell boundary')
        ax6.plot(BuyBoundary,color='darkgreen',linewidth=2,label='Buy boundary')
        ax6.set_ylabel(r'risky fraction $\pi$')
        ax6.set_xlabel(r'time t')
        '''                  
        
        ax8 = fig.add_subplot(338)
        ax8.title.set_text(r'values of $\hat{\sigma}$')
        ax8.set_xlabel(r'time t')
        
        x_axes2 = np.arange(self.N)/(self.N)
        ax9 = fig.add_subplot(339)
        ax9.title.set_text('mean loss contribution over time')
        ax9.set_xlabel(r'time step')
        ax9.plot(x_axes2,LOSS,color='navy',label = r'$\theta^{2}_{t_{n}}h$')
       # ax2.plot(S[0,:,:],color='black',label='shadow price')
       # ax2.plot((1+self.gamma)*P[0,:,:],color='darkgreen',label='bid price')  
       # ax2.plot((1-self.gamma)*P[0,:,:],color='darkred',label='ask price')
    

                
        for i in range(M):
            
            ax1.plot(x_axes,X[i,:,:])
            ax5.plot(x_axes,self.drift_val[i,:,:])
            ax6.plot(x_axes,self.diffusion_val[i,:,:])
            ax7.plot(x_axes,A[i,:,:])
            ax8.plot(x_axes,B[i,:,:])
    
            if i <1:
                ax3.plot(x_axes,PI_t[i,:,:],ls = '--',color='grey',label='optimal fraction')
                ax2.plot(x_axes,SB[i,:,:],ls = '--',color='black',label='strategy bank account')
                ax2.plot(x_axes,SR[i,:,:],ls = '--',color='green',label='strategy risky asset')
              #  ax6.plot(x_axes,Y[i,:,:],ls = '--',color='darkgrey')
                ax4.plot(x_axes,Y[i,:,:],ls = '--')
            else:
              #  ax6.plot(x_axes,Y[i,:,:],ls = '--',color='darkgrey')
                ax4.plot(x_axes,Y[i,:,:],ls = '--')
                ax3.plot(x_axes,PI_t[i,:,:],ls = '--',color='grey')
                ax2.plot(x_axes,SB[i,:,:],ls = '--',color='black')
                ax2.plot(x_axes,SR[i,:,:],ls = '--',color='green')
        
        ax3.plot(x_axes,SellBoundary,color='darkred',linewidth=3,label='Sell boundary')
        ax3.plot(x_axes,BuyBoundary,color='darkgreen',linewidth=3,label='Buy boundary')
       
        ax4.plot(x_axes,torch.ones_like(Y[0,:,:]),linewidth=3,color='black')
        ax4.plot(x_axes,torch.ones_like(Y[0,:,:])*self.s.detach(),linewidth=3,color='black',label='Interval boundaries of $[1,\hat{s}]$')
        
        
        ax4.legend()
        ax3.legend()
        ax2.legend()
        ax9.legend()
        
        plt.show()

        print('For initial capital %.4f the expected utility of the terminal wealth is %.4e'%(self.x,ETW))
        
        if save:
            if heat_path is not None:
                fig.savefig(heat_path + '/plot_{}.jpg'.format(self.stamp))
            else:
                fig.savefig('./results/%s/full_plots_%i.jpg'%(name,M))
            
        return
    
    def plot_strategy_and_fraction(self,M,save=False, name='shadow_price'):
        
        X, S, P, ETW, PI, PI_t, SB, SR, LOSS, A, B, Y = self.predict(M)
        
        x_axes = np.arange(self.N+1)/(self.N+1)
        fig = plt.figure(figsize=(15,5))
        fig.suptitle('%i simulations of learned processes - initial capital: %.1f - eotw: %.4e'%(M,self.x,ETW),fontsize=14)
        
        ax1 = fig.add_subplot(121)
        ax1.title.set_text('optimal strategy')
        ax1.set_ylabel('strategy')
        ax1.set_xlabel(r'time t') 
        
        ax2 = fig.add_subplot(122)
        ax2.title.set_text(r'values of $\pi$ with transaction costs')
        ax2.set_ylabel(r'risky fraction $\pi$')
        ax2.set_xlabel(r'time t')
        
        for i in range(M):
            if i <1:
                ax2.plot(x_axes,PI_t[i,:,:],ls = '--',color='grey',label='optimal fraction')
                ax1.plot(x_axes,SB[i,:,:],ls = '--',color='black',label='strategy bank account')
                ax1.plot(x_axes,SR[i,:,:],ls = '--',color='green',label='strategy risky asset')
            else:
                ax2.plot(x_axes,PI_t[i,:,:],ls = '--',color='grey')
                ax1.plot(x_axes,SB[i,:,:],ls = '--',color='black')
                ax1.plot(x_axes,SR[i,:,:],ls = '--',color='green')
        
        ax2.plot(x_axes,SellBoundary,color='darkred',linewidth=3,label='Sell boundary')
        ax2.plot(x_axes,BuyBoundary,color='darkgreen',linewidth=3,label='Buy boundary')
       
        ax1.legend()
        ax2.legend()
        
        if save:
            fig.savefig('./results/%s/strategy_fraction_%i.jpg'%(name,M))
            
        return
    def plot_training_loss(self,save=False):
        
        Y = self.training_loss
        Iters = len(Y)
        X = np.arange(Iters)
        
        fig = plt.figure(figsize=(16,4))
        
        fig.suptitle('Training error with learning rates %s and Iter=%s, lrerror= %.4f'%(str(self.used_lr),str(self.used_iter),self.training_loss[-1]))
        
        plt.plot(X, Y)
        plt.xlabel('Iter')
        plt.ylabel('expected terminal wealth',rotation='vertical')
        
        if save:
            fig.savefig('./plots/%s.jpg'%(self.stamp))
        return 
