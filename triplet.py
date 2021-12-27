import numpy as np
import torch
import math


def bm_triplet(x,d=0.0,v=0.4,T=1,X=None,is_torch=True):
    
    x = np.array(x)
    X = np.array(X)
    
    mu = np.sqrt(T)*(d/v)
    
    if X is None:
        X = np.random.normal(size=x.shape)   
        
    z = X + mu
    a = min_value(z)
    b = max_value(z,a)
    
    W_x = x+np.sqrt(T)*v*z
    m = x+np.sqrt(T)*v*a
    M = x+np.sqrt(T)*v*b
    
    if is_torch:
        return torch.tensor(W_x,dtype=torch.float),torch.tensor(m,dtype=torch.float),torch.tensor(M,dtype=torch.float)
    
    else:
        return W_x,m,M

def min_value(z):
    
    u = np.random.uniform(size=z.shape)
    a = z/2 - np.sqrt((z**2)/4 - np.log(u)/2)
    
    return a

def max_value(z,a,eps=1e-12):
    
    u = np.random.uniform(size=z.shape)
    l = relu(z)
    r = np.ones_like(z)*float('inf')
    
  #  print('z',z.shape)
  #  print('a',a.shape)
  #  print('u',u.shape)
  #  print('r',r.shape)
    xn = relu(np.log(1/(relu(z) - a)))/2 + z/2 + np.sqrt(((z**2)/4)-((1+np.log(1+((a**2)/(np.abs(z)+0.01))))*np.log(1-u))/2)
    x0 = np.ones_like(xn)*(float('inf'))
    
    while (np.any(np.abs(xn-x0))>=eps):
#    while np.any((xn+x0)/2 <= z):
        
        x0 = xn
        f = pdf_max_value(x0,z,a)
        F = cdf_max_value(x0,z,a)
        
        maskF = F<u
        
        l[maskF] = x0[maskF]
        r[~maskF] = x0[~maskF]
                
        mask_ub = (xn >= r)
        mask_lb = (xn <= l)
        maskr = r < float('inf')
        
        xn[mask_ub*maskr] = ((l+r)/2)[mask_ub*maskr]
        xn[mask_lb*maskr] = ((l+r)/2)[mask_lb*maskr]
        
        xn[mask_ub*(~maskr)] = (1.2*(l+0.1))[mask_ub*(~maskr)]
        xn[mask_lb*(~maskr)] = (1.2*(l+0.1))[mask_lb*(~maskr)]

    return (xn+x0)/2

def pdf_max_value(b,z,a):
    
    C = np.exp((2*a*(a-z))/(z-2*a))
    S = np.zeros_like(b)
    dS = np.ones_like(b)*float('inf')
    
    i=1
    while np.any(dS > 1e-12):

        dS = summand_pdf_max_value(z,a,b,i) + summand_pdf_max_value(z,a,b,-i)
        S += dS
        i+=1
        
    return S

def summand_pdf_max_value(z,a,b,k):
    S1 = -2*(k**2)*(1-(z+2*k*(b-a))**2)*np.exp(-(((z+2*k*(b-a))**2)/2)+((z**2)/2))
    S2 = 2*k*(k+1)*(1-(z-2*a+2*k*(b-a))**2)*np.exp(-((z-2*a+2*k*(b-a))**2)/2+((z**2)/2))
    return S1 + S2

def cdf_max_value(b,z,a):
    
    C = np.exp((2*a*(a-z))/(z-2*a))
    S = np.zeros_like(b)
    dS = summand_cdf_max_value(z,a,b,0)
    
    i=1
    while np.any(np.abs(dS) > 1e-12):

        dS = summand_cdf_max_value(z,a,b,i) + summand_cdf_max_value(z,a,b,-i)
        S += dS
        i+=1
        
    return S

def summand_cdf_max_value(z,a,b,k):
    S1 = (k+1)*(z-2*a+2*k*(b-a))*np.exp(-(((z-2*a+2*k*(b-a))**2)/2)+((z**2)/2))
    S2 = -k*(z+2*k*(b-a))*np.exp(-(((z+2*k*(b-a))**2)/2)+((z**2)/2))
    return S1 + S2

def relu(X):
    
    x = X.copy()
    x[x<0] = 0
    
    return x
