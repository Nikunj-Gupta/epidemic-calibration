import gym, numpy as np, pandas as pd, argparse, matplotlib.pyplot as plt 

from gym import spaces 

from stable_baselines3 import PPO 
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList 
from stable_baselines3.common.env_checker import check_env 


from scipy.integrate import odeint 
from sklearn.metrics import mean_absolute_error 

parser = argparse.ArgumentParser(description='Optional app description') 
parser.add_argument('--total_timesteps', type=int, default=100) 
parser.add_argument('--max_ep_len', type=int, default=10) 
parser.add_argument('--done_reward', type=int, default=0) 
parser.add_argument('--norm_reward', type=int, default=1_000_000) 
parser.add_argument('--num_params', type=int, default=2) # SIR 
parser.add_argument('--eval_trials', type=int, default=10) 
parser.add_argument('--save_freq', type=int, default=1000) 

args = parser.parse_args()

TOTAL_TIMESTEPS = args.total_timesteps 
MAX_EP_LENGTH = args.max_ep_len 
DONE_REWARD = args.done_reward  
NORM_REWARD = args.norm_reward
NUM_PARAMS = args.num_params 
NUM_EVAL_TRIALS = args.eval_trials
SAVE_FREQ = args.save_freq 
if NUM_PARAMS ==2: PARAMS = ["beta", "gamma"] 
EXPNAME = "--".join([
    "exp", 
    "max_ep_len_" + str(MAX_EP_LENGTH), 
    "done_reward_" + str(DONE_REWARD), 
    "norm_reward_" + str(NORM_REWARD) 
]) 
LOGDIR = "testML_subgroups/ensemble_rl_logs/" + EXPNAME 



"""
Actions 
==> Increase/Decrease by 0.1, 0.01, 0.001 for both parameters + No change 
"""
actions = ["no change"] 
for k in ["beta", "gamma"]: 
    for i in ["increase", "decrease"]: 
        for j in [0.1, 0.01, 0.001]: 
            actions.append(tuple((k, i, j)))
ACT_DIM = len(actions) 

# [print(action) for action in actions] 
# print("Number of actions: ", ACT_DIM) 


"""
Compartmental Epidemic data 
"""
data = pd.read_csv('testML_subgroups/data/sim_175/data_SIR_grp1.csv') 


"""
Config: SIR 
"""
n = 175 
y0 = [299999999, 1, 0] 
t = np.linspace(0, n, n)
N = 3e8 
cor_tab = [[0,1,0], [0,0,1], [0,0,0]] 
nb_comp = 3 
name_comp = ["Suspected", "Infected", "Recovered"] 


def model_deriv(y, t, N, params, cor_tab, nb_comp):
    
        dy = np.zeros(nb_comp)
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
        return dy 

"""
actions 
"""
def calc_params_from_actions(act, beta, gamma): 
    a = actions[act] 
    if isinstance(a, tuple): 
        if a[0]=="beta" and a[1]=="increase":  return beta+a[2]  , gamma 
        if a[0]=="beta" and a[1]=="decrease":  return beta-a[2]  , gamma 
        if a[0]=="gamma" and a[1]=="increase": return beta       , gamma+a[2] 
        if a[0]=="gamma" and a[1]=="decrease": return beta       , gamma-a[2] 
    return beta, gamma 

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, obs_dim, act_dim): 
        super(CustomEnv, self).__init__() 
        self.action_space = spaces.Discrete(act_dim) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) 
        self.observation = None 
        self.timestep = 0 

    def step(self, action): 
        self.timestep += 1 
        
        """
        get model parameters (beta, gamma) from action 
        """
        new_beta, new_gamma = calc_params_from_actions(action, self.observation[0], self.observation[1]) 
        
        
        """
        calculate next state 
        """
        self.observation = np.array([new_beta, new_gamma], dtype=np.float32) 
        
        
        """
        apply action to current state 
        """
        fitted_parameters = (new_beta, new_gamma) 
        res = odeint(model_deriv, y0, t, args=(N, fitted_parameters, cor_tab, nb_comp)) 
        fitted_curve = res.T 
        fitted_curve_dataframe = pd.DataFrame(columns=name_comp) 
        for i in range(nb_comp): fitted_curve_dataframe[name_comp[i]] = fitted_curve[i] 
        mae = mean_absolute_error(data['Infected'], fitted_curve_dataframe['Infected'])  # MAE of simulated infected data points 

        """
        calculate reward
        """
        reward = -1 * mae / NORM_REWARD 
        # reward = -1 * (mae / N) 
        # reward = N / mae 
        
        """
        calculate done and info 
        """ 
        done = True if ((abs(reward) < DONE_REWARD) or (self.timestep >= MAX_EP_LENGTH)) else False 
        info = {
                "beta": new_beta, 
                "gamma": new_gamma, 
                "mae": mae 
            } 
        self.info = info 
        
        return self.observation, reward, done, info 

    def reset(self):
        self.timestep = 0 
        self.observation = np.array([np.random.uniform(0, 1), np.random.uniform(0, 0.05)], dtype=np.float32) 
        return self.observation  

env = CustomEnv(obs_dim=NUM_PARAMS, act_dim=ACT_DIM) 
env.reset() 

# Checking custom environment and output additional warnings if needed 
check_env(env) 

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=SAVE_FREQ,
  save_path=LOGDIR,
  name_prefix="rl_model",
)
# Separate evaluation env
eval_env = CustomEnv(obs_dim=NUM_PARAMS, act_dim=ACT_DIM) 
# Use deterministic actions for evaluation
eval_callback = EvalCallback(
                        eval_env, 
                        best_model_save_path=LOGDIR,
                        log_path=LOGDIR, 
                        eval_freq=500,
                        deterministic=True, 
                        render=False
                    )
# Define and Train the agent
model = PPO("MlpPolicy", 
            env, 
            tensorboard_log=LOGDIR, 
            policy_kwargs=dict(
                            net_arch=[32, 32, 32] 
                        ), 
        )

model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    callback=CallbackList([checkpoint_callback, eval_callback]), 
    tb_log_name="run"
) 

obs = env.reset()
betas = [] 
gammas = [] 
for trial in range(NUM_EVAL_TRIALS):
    for _ in range(MAX_EP_LENGTH):
        action = model.predict(obs, deterministic=True) 
        obs, reward, done, info = env.step(action[0])     
        betas.append(obs[0]) 
        gammas.append(obs[1]) 
        if done: 
            # print(reward)
            # print(info)
            obs = env.reset()

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=NUM_EVAL_TRIALS) 
approx_mae = mean_reward * NORM_REWARD 
print("Approx MAE = ", approx_mae) 