# A Deep Learning Approach to Optimal Investment With Transaction Costs

In [code](./code) you find the PyTorch implementation of the numerical method proposed in my master's thesis [thesis.pdf](./thesis.pdf) to approximate the optimal trading strategy and the optimal risky fraction in a Black-Scholes model with transaction costs.

The results along with the trained neural networks of the experiments of Chapter 4 in [thesis.pdf](./thesis.pdf) are provided in [results_thesis.zip](./code/results/results_thesis.zip).

To run the code and to train your own neural networks, clone this repository and execute the jupyter notebook [showcase.ipynb](./code/showcase.ipynb).

Moreover, you find in [code/processes/ItoProcesses.py](./code/processes/ItoProcesses.py) an implementation to simulate Brownian motion, geometric Brownian motion and the triplet of final, minimal and maximal value of scaled Brownian motion with drift. To simulate a differnet Ito process, you may define 

```python
ItoProcess(Minibatch):
  def __init__(self, T=1, N=100, D=1,s0=0):
    super().__init__(T=T, N=N, D=D,s0=s0)
        
    def increment(self,M):
        #your code to specify the distribution of the increment
        return
    
    def path(self,M):
        return super().path(M)
    
    def step(self,t,x,h,dX):
        return super().step(t,x,h,dX)
    
    def drift(self,t,S):
        #your code to specify the drift function 
        return
    
    def vol(self,t,S):
        #your code to specify the volatility function 
        return 
'''
