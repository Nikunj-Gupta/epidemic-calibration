#wip

import numpy as np, argparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import csv
import random

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--noise_val', type=int, default=11, help='A required integer positional argument') 
parser.add_argument('--gen_datasheet', type=int, default=1, help='A required integer positional argument') 
parser.add_argument('--n', type=int, default=175, help='A required integer positional argument') 
parser.add_argument('--csv', type=int, default=1, help='A required integer positional argument') 
parser.add_argument('--plot', type=int, default=0, help='A required integer positional argument') 
args = parser.parse_args()


N = 3e8 #population

I0 = 1 #Initial number of infected
R0 = 0 #Initial number of recovered
D0 = 0 #Initial number of quarantined
V0 = 0 #Initial number of vaccinated
S0 = N - I0 - R0 - V0 #initial number of susceptible 
y0 = S0, I0, R0, V0, D0 


#parameters (to determine)
beta = 0.75 # contact rate
gamma = 1./10 # recovered rate
delta = 1./100 # chance of dying 
alpha = 1./50 # vaccination rate 
sigma = 1./20 # susceptibility rate 


n=175 #number of days
t = np.linspace(0, n, n) #timeseries (\days)

#SIRVD model
def deriv(y, t, N, beta, gamma, alpha, sigma, delta) : 
    S, I, R, V, D = y0
    dSdt = -beta * S * I / N + sigma * R - alpha * S 
    dIdt = beta * S * I / N - gamma * I - delta * I 
    dRdt = gamma * I - sigma * R 
    dVdt = alpha * S 
    dDdt = delta * I 
    return dSdt, dIdt, dRdt, dVdt, dDdt 

#simple function to add noise
def noise(S,I,R,V,D,val,n,N): 
    for i in range(n):
        noise1 = (val/100)*N*random.random()
        frac1 = random.random()
        frac2 = random.random()
        frac3 = random.random()
        ni = noise1
        ns = ni * frac1
        nr = ni * (1 - frac1)*frac2
        nv = ni * (1 -frac1)*(1-frac2)*frac3
        nx = ni * (1 -frac1)*(1-frac2)*(1-frac3)
        I[i] = I[i] + ni
        S[i] = S[i] - ns
        R[i] = R[i] - nr
        V[i] = V[i] - nv
        D[i] = D[i] - nx
    return abs(S) , I , abs(R) , abs(V), abs(D) 


#using odeint to intigrate and solve
res = odeint(deriv, y0, t, args=(N, beta, gamma, alpha, sigma, delta)) 
S, I, R, V, D = res.T
print(I)


val=args.noise_val
S, I, R, V, D = noise(S,I,R,V,D,val,n,N)
noise_name = "_n"+str(val)

#to generate datasheets
if args.gen_datasheet:
    #txt dile
    size=n 
    print(size)

    if not args.csv: 
        #txt file
        ficname="data_SIRVD_"+str(size)+".txt"
        fic = open("data/"+ficname,"w")
        for i in range(size):
            fic.write(str(i+1)+","+str(int(S[i]))+","+str(int(I[i]))+","+str(int(R[i]))+","+str(int(V[i]))+","+str(int(D[i]))+"\n")
        fic.close()

    else : 
        #csv file
        header = ['Day','Suspected', 'Infected','Recovered','Vaccinated', 'Death']
        ficnamecsv="data_SIRVD_"+str(size)+noise_name+".csv"

        with open("data/"+ficnamecsv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(size):
                data = [int(i+1),int(S[i]),int(I[i]),int(R[i]),int(V[i]),int(D[i])]
                writer.writerow(data)
    print("Data succesfully created")


#display
if args.plot: 
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(t, V, 'm', alpha=0.5, lw=2, label='Vaccinated')
    ax.plot(t, D, 'y', alpha=0.5, lw=2, label='Death')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number ')
    ax.set_ylim(0, N*1.1)
    ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
