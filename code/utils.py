# +
import torch
import matplotlib.pyplot as plt

def approx_m(s,SP,m,m1,M1):
    
    '''
    Algorithm 1:
        Approximation of process defined in Definition 8; Al
    '''
        
    value0, cmask0, pmask0 = m
    value1, cmask1, pmask1 = torch.ones_like(value0), torch.ones_like(cmask0), torch.ones_like(pmask0)  

    value1[cmask0] = m1[cmask0]
    pmask1[cmask0] = True
    value1[~cmask0] = (M1[~cmask0]/s)
    pmask1[~cmask0] = False

    min_to_min = cmask0*pmask0
    min_to_min_larger = (value1[min_to_min]>value0[min_to_min])
    value1[min_to_min][min_to_min_larger] = value0[min_to_min][min_to_min_larger]

    max_to_max = (~cmask0)*(~pmask0)
    max_to_max_smaller = (value1[max_to_max]<value0[max_to_max])
    value1[max_to_max][max_to_max_smaller] = value0[max_to_max][max_to_max_smaller]

    not_hitting_ub = (SP/value1)<s
    cmask1[cmask0*not_hitting_ub] = True
    cmask1[cmask0*(~not_hitting_ub)] = False

    not_hitting_low = (SP/(s*value1))>(1/s)
    cmask1[(~cmask0*not_hitting_low)] = False
    cmask1[(~cmask0*(~not_hitting_low))] = True

    return (value1, cmask1, pmask1) 


# -

def plot_learning_summary(loss,sell_boundary,buy_boundary,points_of_interst=[0,1,2,3],save=False,name='shadow_price'):
        
        fig = plt.figure(figsize=(16, 10))
        ax1 = fig.add_subplot(2,3,1) 
        ax1.title.set_text('1000 realizations of initialized $\hat{\pi}$')
        ax1.set_xlabel('Time $t$')
        ax1.set_ylabel(r'risky fraction $\pi$')
        
        ax2 = fig.add_subplot(2,3,2)
        ax2.title.set_text('1000 realizations of intermediate $\hat{\pi}$')
        ax2.set_xlabel('Time $t$')
        
        ax3 = fig.add_subplot(2,3,3)
        ax3.title.set_text('1000 realizations of learned $\hat{\pi}$')
        ax3.set_xlabel('Time $t$')
        
        pi0 = points_of_interst[0]
        pi1 = points_of_interst[1]
        pi2 = points_of_interst[2]
        
        for i in range(100):
            ax1.plot(pi0[i,:,:],ls='--',color='grey')
            ax2.plot(pi1[i,:,:],ls='--',color='grey')
            ax3.plot(pi2[i,:,:],ls='--',color='grey')
            
        ax1.plot(sell_boundary,color='darkred',linewidth=3,label='sell boundary')
        ax1.plot(buy_boundary,color='darkgreen',linewidth=3,label='buy boundary')
        ax1.legend()
        
        ax2.plot(sell_boundary,color='darkred',linewidth=3,label='sell boundary')
        ax2.plot(buy_boundary,color='darkgreen',linewidth=3,label='buy boundary')
        ax2.legend()
        
        ax3.plot(sell_boundary,color='darkred',linewidth=3,label='sell boundary')
        ax3.plot(buy_boundary,color='darkgreen',linewidth=3,label='buy boundary')
        ax3.legend()  
        
        ax4 = fig.add_subplot(2,2,(3,4)) # two rows, two colums, combined third and fourth cell
        ax4.set_ylabel(r'empirical loss')
        ax4.set_xlabel(r'optimization step')
        ax4.plot(loss)

        if save:
            fig.savefig('./results/%s/learning_summary.jpg'%(name))
        return

