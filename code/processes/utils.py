import matplotlib.pyplot as plt
import numpy as np
import time
from ItoProcesses import gBmTriplet

def plot_triplet(Wn=None, T=1, N=100, D=1, p0=1, mu=0.08, sigma=0.4,save=False):
        
        if Wn==None:
            Wn = gBmTriplet(T=T, N=N, D=D, mu=mu, sigma=sigma)
            
        W, rm, rM, S, erm, erM, P = Wn.full_path(1)
        N = Wn.N
        
        fig  = plt.figure(figsize=(16,4))
        
        ax1 = fig.add_subplot(121)
        ax1.set_title('A realization of $\mathcal{W}_{N}$',fontsize=20)
        ax1.set_xlabel('Time $t$')
        ax1.plot(np.arange(N+1)/(N+1),W[0,:,:],color='mediumblue',label='Scaled Brownian motion with drift $W$')
        ax1.plot(np.arange(N+1)/(N+1),rM[0,:,:],color='forestgreen',label='Maximum Process $M$ on $[t_{n},t_{n+1}]$') 
        ax1.plot(np.arange(N+1)/(N+1),rm[0,:,:],color='darkorange',label='Minimum Process $m$ on $[t_{n},t_{n+1}]$')
        plt.legend(fontsize=10)
        
        ax2 = fig.add_subplot(122)
        ax2.set_title('A realization of $\exp(\mathcal{W})_{N}$',fontsize=20)
        ax2.set_xlabel('Time $t$')
        ax2.plot(np.arange(N+1)/(N+1),S[0,:,:],color='mediumblue',label='$\exp(W)$')
        ax2.plot(np.arange(N+1)/(N+1),erM[0,:,:],color='forestgreen',label='$\exp(M)$') 
        ax2.plot(np.arange(N+1)/(N+1),erm[0,:,:],color='darkorange',label='$\exp(m)$')
        plt.legend(fontsize=10)
        
        plt.subplots_adjust(wspace=0.1)
        plt.show()
        if save:
            fig.savefig('./plots/triplet/Triplet_%s.jpg'%(str(time.time() - int(time.time()))[2:7]))
        return
