import os 
from pathlib import Path 
from itertools import count 

dumpdir = "./runs/" 
if not os.path.isdir(dumpdir):
    os.mkdir(dumpdir)
fixed_text = "#!/bin/bash\n"\
             "#SBATCH --nodes=1\n"\
             "#SBATCH --cpus-per-task=16 \n"\
             "#SBATCH --time=24:00:00\n"\
             "#SBATCH --mem=40GB\n"

for noise_val in range(1, 16): 
    for model in ['sirvd']: 
        command = fixed_text 
        command += "\nsource ../venvs/calibration/bin/activate\n"\
            "\nmodule load python/intel/3.8.6\n"\
            "module load openmpi/intel/4.0.5\n"\
            "time python3 minimize_test.py " 
        command = ' '.join([
            command, 
            '--model', model, 
            '--start 27', 
            '--end 72', 
            '--step 1', 
            '--basename n'+str(noise_val), 
            ]) 
        # print(command) 
        log_dir = Path(dumpdir)
        for i in count(1):
            temp = log_dir/('run{}.sh'.format(i)) 
            if temp.exists():
                pass
            else:
                with open(temp, "w") as f:
                    f.write(command) 
                log_dir = temp
                break 
