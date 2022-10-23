import argparse 
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--model', type=str, default="SIR", help='A required integer positional argument') 
parser.add_argument('--noise_val', type=int, default=1, help='A required integer positional argument') 
args = parser.parse_args()
noise_val = args.noise_val 
ficname="config_"+args.model+"_n"+str(noise_val)+".txt"
fic = open("data/"+ficname,"w") 
if args.model == 'SIR': 
    fic.write( "3e8\n1\n0\n" + "data_SIR_175_n"+str(noise_val)+".csv\n" + "Infected\nInfected\nInfected\n37\n0.3\n0.1") 
elif args.model == 'SIRD':
    fic.write( "3e8\n1\n0\n0\n" + "data_SIRD_175_n"+str(noise_val)+".csv\n" + "Infected\nDeath\n34\n0.3\n0.1\n0.02") 
fic.close() 


