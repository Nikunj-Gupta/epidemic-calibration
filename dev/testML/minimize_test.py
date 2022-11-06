from matplotlib.font_manager import json_dump
from minimize import SIRModel, SIRDModel, SIRVDModel 
import matplotlib.pyplot as plt, pandas as pd, numpy as np, argparse 
from sklearn.metrics import mean_squared_error 

def disp(train_size, t, fitted_parameters, fit, data_nr, mae, methods):	
    fig, ax = plt.subplots(figsize=(8.26, 8.26))
    ax.set_ylim(0,data_nr.max()*1.1)
    ax.set_title('Infected')
    plt.axvline(x=train_size,color='gray',linestyle='--', label="End of train dataset")
    ax.scatter(t, data_nr, marker='+', color='black', label=f'Measures (method = {methods})')
    ax.plot(t, fit[:][1], 'g-', label=f'Simulation')
    ax.vlines(t, data_nr, fit[:][1], color='g', linestyle=':', label=f'MAE = {mae:.1f}')
    fig.legend(loc='upper center')
    plt.show()
    plt.close(fig)
    
def change_train_size(config_name, train_size, model='SIR'):  
    with open("data/"+config_name, "r") as f:
        data = f.readlines()
    f.close()
    key = None 
    if model=='SIR' or model=='SIRD': 
        key = 7 
    elif model=='SIRVD': 
        key = 8 
    data[key] = data[key].rstrip("\n")
    f = open("data/"+config_name, "r")
    replacement = ""
    for line in f:
        line = line.strip()
        changes = line.replace(data[key], str(train_size))
        replacement = replacement + changes + "\n"
    f.close()
    fout = open("data/"+config_name, "w")
    fout.write(replacement)
    fout.close()

def data_set(data, train_size, methods, fitted_parameters, mae, model='SIRVD'):
    data['Starting_Days'].append(train_size)
    data['Methods'].append(methods)
    data['Beta'].append(fitted_parameters[0])
    data['Gamma'].append(fitted_parameters[1])
    if model=='SIR': 
        data['Kappa'].append(None) 
        data['Alpha'].append(None)
        data['Sigma'].append(None)
        data['Delta'].append(None)
        gt_vector = [0.5, 0.1] 
    elif model=='SIRD': 
        data['Kappa'].append(fitted_parameters[2]) 
        data['Alpha'].append(None)
        data['Sigma'].append(None)
        data['Delta'].append(None)
        gt_vector = [0.75, 0.1, 0.01] 
    elif model=='SIRVD': 
        data['Alpha'].append(fitted_parameters[2])
        data['Sigma'].append(fitted_parameters[3])
        data['Delta'].append(fitted_parameters[4])
        data['Kappa'].append(None) 
        gt_vector = [0.75, 0.1, 0.01, 0.02, 0.05] 
    data['mse'] = mean_squared_error(gt_vector, fitted_parameters) 
    data['Mae'].append(mae)

    return data

def data_frame(data, start, end, step, basename, modelname):
    print(data)
    df = pd.DataFrame(data)
    df.to_csv(f'data/{modelname}_{basename}.csv', sep = ',')
    for j in range(start,end,step):
        dfOpt = df[ df.Starting_Days == j ]
        print('Most effective method:')
        print(df.iloc[dfOpt.Mae.idxmin(),:])
    return df

def final_plot(df, basename, modelname):
    fig, ax = plt.subplots(figsize=(8.26,8.26))
    ax.set_title(f'Comparison of methods ({modelname} model, {basename})', fontsize=20)
    for method in df.Methods.unique():
        _df = df[ df.Methods == method]
        ax.plot(_df.Starting_Days.apply(lambda v: str(v)), _df.Mae, label=method)
    ax.set_xlabel('Starting day', fontsize=16)
    ax.set_ylabel('MAE', fontsize=16)
    #~ ax.set_yscale('linear')
    #~ ax.set_ylim([1.3e7, 1.5e7])
    ax.legend(ncol=2, loc='upper center', fontsize=16)
    #~ plt.grid()
    plt.savefig(f'fig/fig_{modelname}_{basename}.pdf')
    # plt.show()
    plt.close(fig)

#---------------- Fitted on NYC data ---------------------#
#Creating the SIRModel

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--model', type=str, default='sir', help='A required integer positional argument') 
parser.add_argument('--start', type=int, default=27, help='A required integer positional argument') 
parser.add_argument('--end', type=int, default=38, help='A required integer positional argument') 
parser.add_argument('--step', type=int, default=1, help='A required integer positional argument') 
parser.add_argument('--basename', type=str, default="n11", help='A required integer positional argument') 
parser.add_argument('--plot', type=int, default=1, help='A required integer positional argument') 
args = parser.parse_args()

if args.model.upper() == 'SIR': 
    model = SIRModel() 
elif args.model.upper() == 'SIRD': 
    model = SIRDModel() 
elif args.model.upper() == 'SIRVD': 
    model = SIRVDModel() 

methods=["leastsq",'least_squares','differential_evolution','brute','basinhopping','ampgo','nelder','lbfgsb','powell','cg','cobyla','bfgs','trust-constr','tnc','slsqp','shgo','dual_annealing']
# methods=["leastsq",'least_squares']
# methods=["leastsq"]

# data = {'Starting_Days': [], 'Methods': [], 'Beta': [], 'Gamma': [], 'Mae': []}
data = {'Starting_Days': [], 'Methods': [], 'Beta': [], 'Gamma': [], 'Alpha': [], 'Kappa': [], 'Sigma': [], 'Delta': [], 'Mae': []}
start, end, step, basename = args.start, args.end+1, args.step, args.basename 
configFile = f'config_{args.model.upper()}_{basename}.txt'

for method in methods:
    for j in range(start,end,step):
        change_train_size(configFile, j, model=args.model.upper())
        out, data_nr, fitted_parameters, fit, mae, t = model.fit(configFile , method)
        # data2 = data_nr[1,:]
        #~ disp(j, t, fitted_parameters, fit, data2, mae, method)
        
        data = data_set(data, j, method, fitted_parameters, mae, model=args.model.upper())

df = data_frame(data, start, end, step, basename, modelname=args.model.upper())
if args.plot: 
    final_plot(df, basename, modelname=args.model.upper()) 