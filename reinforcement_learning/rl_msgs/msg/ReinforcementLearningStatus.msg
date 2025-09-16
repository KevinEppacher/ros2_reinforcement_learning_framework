std_msgs/Header header

# IDs
string task
string model_name
string run_id
string stage          # "train" | "eval"
string device         # "cpu" | "cuda"
uint32 seed
uint32 n_envs

# Progress
uint64 total_timesteps
uint64 timesteps
float32 progress_pct
float32 sps
builtin_interfaces/Duration eta

# Episode metrics
uint32 episode
float32 episode_reward
float32 episode_reward_mean
float32 episode_reward_std
float32 best_mean_reward

# Observation/action config
uint32 frame_stack
uint32 obs_h
uint32 obs_w
uint32 obs_c
bool normalize_obs
string action_space
string obs_space

# Optimizer/losses
float32 learning_rate
float32 policy_loss
float32 value_loss
float32 entropy

# Artifacts
string logs_dir
string models_dir
string videos_dir
string last_checkpoint
string last_video
string tensorboard_dir