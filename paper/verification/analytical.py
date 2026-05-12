import numpy as np

L = 50
x = np.linspace(0,L,50)

N = np.arange(1, 20001, 2)
t = 20
alpha = 1.0
kappa = alpha**2

def u_t(x, t, kappa, N):
    
    def term(x, t, kappa, n):
        return 1/n*np.exp(-n**2*np.pi**2*kappa*t/2500)*np.sin(n*np.pi*x/50)
    
    solution = np.zeros(len(x))
    for n in N:
        temp = term(x, t, kappa, n)
        solution += temp
    solution *= 80/np.pi
    return solution

initial_condition = u_t(x,t,kappa,N)
print(initial_condition)


import matplotlib.pyplot as plt
plt.plot(x, initial_condition)
plt.savefig("initial.pdf")