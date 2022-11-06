import pandas as pd, glob, argparse 


def data_frame(data_filepath, start, end, step, noise, num_compartments, model_name, ml_df):
    df = pd.read_csv(data_filepath, sep = ',') 
    for j in range(start, end, step):
        dfOpt = df[ df.Starting_Days == j ]
        # print('Most effective method:') 
        # print(df.iloc[dfOpt.Mae.idxmin(),:]) 
        # most_effective_method = df.iloc[dfOpt.Mae.idxmin(),:] 
        effective_methods = df.iloc[dfOpt.Mae.nsmallest(3, keep='all').index,:] 
        most_effective_method, second_most_effective_method, third_most_effective_method = effective_methods.iloc[0], effective_methods.iloc[1], effective_methods.iloc[2] 

        ml_df.loc[len(ml_df.index)] = [ 
            most_effective_method['Starting_Days'], 
            noise, 
            num_compartments, 
            most_effective_method['Mae'], 
            most_effective_method['Methods'], 
            second_most_effective_method['Mae'], 
            second_most_effective_method['Methods'], 
            third_most_effective_method['Mae'], 
            third_most_effective_method['Methods'], 
            model_name 
        ] 
    return df 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--start', type=int, default=27, help='A required integer positional argument') 
    parser.add_argument('--end', type=int, default=38, help='A required integer positional argument') 
    parser.add_argument('--step', type=int, default=1, help='A required integer positional argument') 
    args = parser.parse_args()

    start, end, step = args.start, args.end+1, args.step 
    ml_df = pd.DataFrame(columns=
        [
            'amount_of_data', 
            'noise', 
            'num_compartments', 
            'best_mae', 
            'best_calibration_method', 
            'second_best_mae', 
            'second_best_calibration_method', 
            'third_best_mae', 
            'third_best_calibration_method', 
            'model_name'
        ]) # Amount_of_Data = Starting_Days 

    # df = data_frame("dev/testSIR/data/sir_n5.csv", start=27, end=37, step=1, noise=5, num_compartments=3, ml_df=ml_df) 
    # df = data_frame("dev/testSIR/data/sir_n10.csv", start=32, end=38, step=1, noise=10, num_compartments=3, ml_df=ml_df) 
    # df = data_frame("dev/testSIRD/data/sird_n5.csv", start=27, end=37, step=1, noise=5, num_compartments=4, ml_df=ml_df) 
    # df = data_frame("dev/testSIRD/data/sird_n10.csv", start=32, end=38, step=1, noise=10, num_compartments=4, ml_df=ml_df) 
    # print(ml_df) 
    
    for name in glob.glob('data/SIR_n*'):
        df = data_frame(name, start=start, end=end, step=step, noise=int(name.split('/')[-1].split('.')[0].split('_')[-1][1:]), num_compartments=3, model_name="SIR", ml_df=ml_df) 
    for name in glob.glob('data/SIRD_n*'):
        df = data_frame(name, start=start, end=end, step=step, noise=int(name.split('/')[-1].split('.')[0].split('_')[-1][1:]), num_compartments=4, model_name="SIRD", ml_df=ml_df) 
    for name in glob.glob('data/SIRVD_n*'):
        df = data_frame(name, start=start, end=end, step=step, noise=int(name.split('/')[-1].split('.')[0].split('_')[-1][1:]), num_compartments=5, model_name="SIRVD", ml_df=ml_df) 
    print(ml_df.shape) 
    ml_df.to_csv("./data/ml_df.csv", index=False) 