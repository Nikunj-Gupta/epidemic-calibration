import pandas as pd, glob, argparse 


def data_frame(data_filepath, start, end, step, noise, num_compartments, ml_df):
    df = pd.read_csv(data_filepath, sep = ',') 
    for j in range(start, end, step):
        dfOpt = df[ df.Starting_Days == j ]
        # print('Most effective method:')
        # print(df.iloc[dfOpt.Mae.idxmin(),:]) 
        most_effective_method = df.iloc[dfOpt.Mae.idxmin(),:] 
        ml_df.loc[len(ml_df.index)] = [most_effective_method['Starting_Days'], noise, num_compartments, most_effective_method['Mae'], most_effective_method['Methods']] 
    return df 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--start', type=int, default=27, help='A required integer positional argument') 
    parser.add_argument('--end', type=int, default=38, help='A required integer positional argument') 
    parser.add_argument('--step', type=int, default=1, help='A required integer positional argument') 
    args = parser.parse_args()


    start, end, step = args.start, args.end+1, args.step 
    
    ml_df = pd.DataFrame(columns=['amount_of_data', 'noise', 'num_compartments', 'best_mae', 'best_calibration_method']) # Amount_of_Data = Starting_Days 

    # df = data_frame("dev/testSIR/data/sir_n5.csv", start=27, end=37, step=1, noise=5, num_compartments=3, ml_df=ml_df) 
    # df = data_frame("dev/testSIR/data/sir_n10.csv", start=32, end=38, step=1, noise=10, num_compartments=3, ml_df=ml_df) 
    # df = data_frame("dev/testSIRD/data/sird_n5.csv", start=27, end=37, step=1, noise=5, num_compartments=4, ml_df=ml_df) 
    # df = data_frame("dev/testSIRD/data/sird_n10.csv", start=32, end=38, step=1, noise=10, num_compartments=4, ml_df=ml_df) 
    # print(ml_df) 

    for name in glob.glob('data/sir_n*'):
        df = data_frame(name, start=start, end=end, step=step, noise=int(name.split('/')[-1].split('.')[0].split('_')[-1][1:]), num_compartments=3, ml_df=ml_df) 
    for name in glob.glob('data/sird_n*'):
        df = data_frame(name, start=start, end=end, step=step, noise=int(name.split('/')[-1].split('.')[0].split('_')[-1][1:]), num_compartments=4, ml_df=ml_df) 
    print(ml_df)
