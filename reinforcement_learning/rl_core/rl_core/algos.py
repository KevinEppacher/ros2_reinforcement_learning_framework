from stable_baselines3 import PPO, A2C, DQN, SAC, TD3

ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "sac": SAC,
    "td3": TD3,
}

def make_algo(name: str):
    key = name.lower()
    if key not in ALGOS:
        raise ValueError(f"Unknown algo '{name}'")
    return ALGOS[key]

ALGO_DEFAULTS = {
    "ppo": {
        "learning_rate": 3.0e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": -1.0,
        "stats_window_size": 100,
        "seed": 0,
        "tensorboard": True,
        "policy_kwargs": "{}"
    },
    "a2c": {
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.0,
        "vf_coef": 0.25,
        "max_grad_norm": 0.5,
        "rms_prop_eps": 1e-5,
        "use_rms_prop": True,
        "normalize_advantage": True,
        "seed": 0,
        "tensorboard": True,
        "policy_kwargs": "{}"
    },
    "dqn": {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.05,
        "seed": 0,
        "tensorboard": True,
        "policy_kwargs": "{}"
    },
    "sac": {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "seed": 0,
        "tensorboard": True,
        "policy_kwargs": "{}"
    },
    "td3": {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 2,
        "seed": 0,
        "tensorboard": True,
        "policy_kwargs": "{}"
    }
}

def get_algo_defaults(algo: str):
    algo = algo.lower()
    if algo not in ALGO_DEFAULTS:
        raise ValueError(f"No defaults for algo '{algo}'")
    return ALGO_DEFAULTS[algo].copy()