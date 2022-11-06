NOISE = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15  

START = 27  
END = 72 
STEP = 1 

gen_data: 
	# $(foreach var,$(NOISE), python SIR_odeint_api.py --noise_val $(var);) 
	# $(foreach var,$(NOISE), python SIRD_odeint_api.py --noise_val $(var);) 
	$(foreach var,$(NOISE), python SIRVD_odeint_api.py --noise_val $(var);) 

gen_config: 
	# $(foreach var,$(NOISE), python gen_configs.py --model SIR --noise_val $(var);)
	# $(foreach var,$(NOISE), python gen_configs.py --model SIRD --noise_val $(var);)
	$(foreach var,$(NOISE), python gen_configs.py --model SIRVD --noise_val $(var);)


fit: 
	# $(foreach var,$(NOISE), python minimize_test.py --model sir --start $(START) --end $(END) --step $(STEP) --basename n$(var) --plot 1;) 
	# $(foreach var,$(NOISE), python minimize_test.py --model sird --start $(START) --end $(END) --step $(STEP) --basename n$(var) --plot 1;) 
	$(foreach var,$(NOISE), python minimize_test.py --model sirvd --start $(START) --end $(END) --step $(STEP) --basename n$(var) --plot 1;) 

pipeline: gen_data gen_config fit 

ml:  
	python ml.py --start $(START) --end $(END) --step $(STEP) 

gen_runs: 
	clear 
	python gen_runs.py 

