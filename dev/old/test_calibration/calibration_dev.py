import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters
from sklearn.metrics import mean_absolute_error
from data_load import dataModel

'''
Reste a faire :
    - faire une sortie graphique ?
    - sortie avec un json ?
'''


#-----------Calibration-Model--------------#

class calibModelDev:

    def __init__(self):
        pass


    def model_deriv(self, y, t, N, params, cor_tab, nb_comp):
    
        dy = np.zeros(nb_comp)
        dy[0]=0
        ind=0
        for i in range(nb_comp):
            for j in range(nb_comp):
                if cor_tab[i][j]==1: 
                    if ((i==0) and (j==1)):
                        dy[i]=dy[i]-(params[ind]*y[i]*y[1])/N
                        dy[j]=dy[j]+(params[ind]*y[i]*y[1])/N
                    else:
                        dy[i]=dy[i]-params[ind]*y[i]
                        dy[j]=dy[j]+params[ind]*y[i]
                    ind+=1
        print(dy[0])
        return dy

    
    def solve(self, y0, t, N, params, cor_tab, nb_comp):

        res = odeint(self.model_deriv, y0, t, args=(N, params, cor_tab, nb_comp)) 
        result = res.T

        return result

    
    def err(self, params, t,  data, y0,  N, cor_tab, nb_comp, name_comp, name_params, nb_params, fit_tab):

        params_deriv = np.zeros(nb_params)
        for i in range(nb_params):
            params_deriv[i]=params[name_params[i]]

        result  = self.solve(y0, t, N, params_deriv, cor_tab, nb_comp)

        d = np.zeros(len(t))
        err = 0
        for i in range(nb_comp):
            d = np.resize(data[name_comp[i]],len(t))
            if fit_tab[i]==1:
                err = pow((result[i,:] - d),2)

        return err

    def calib(self, name_json, train_size, set_gamma=False, method='leastsq', max_nfev=1000):

        d = dataModel()
        guess, N, t, data, y0, cor_tab, nb_comp, name_comp, name_params, fit_tab = d.load_config(name_json)

        t_train= np.linspace(0, train_size, train_size)
        nb_params = len(guess)

        #defining parameters
        params = Parameters() 
        for i in range(nb_params):
            params.add(name_params[i], value=guess[i], min=0, max=10, vary=True)
        params.add('N', value=N, vary=False)
         

        if set_gamma:
            params.add('Gamma', value=1, vary=False)

        #applying the fit
        out = minimize(self.err, params, method=method, args=(t_train, data, y0, N, cor_tab, nb_comp, name_comp, name_params, nb_params, fit_tab),max_nfev=max_nfev)

        fitted_parameters = np.zeros(nb_params)
        for i in range(nb_params):
            fitted_parameters[i]=out.params[name_params[i]].value

        #fitted_parameters=(1.08,1)
        fitted_curve = self.solve(y0, t, N, fitted_parameters, cor_tab, nb_comp)

        return out, fitted_curve, data, name_comp
