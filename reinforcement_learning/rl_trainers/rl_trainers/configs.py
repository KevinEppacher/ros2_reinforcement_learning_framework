from dataclasses import dataclass
from typing import Any, Dict

# ---------------------------- Config types ----------------------------

@dataclass
class MetaCfg:
    namespace: str
    mode: str                 # "train" | "eval"
    task: str
    artifacts_dir: str

@dataclass
class AlgoCfg:
    name: str                 # ppo | a2c | dqn | sac | td3
    device: str               # "cpu" | "cuda" | "auto"
    policy: str               # "MlpPolicy" | "CnnPolicy" ...

@dataclass
class TrainCfg:
    total_timesteps: int
    save_freq: int
    progress: bool
    n_envs: int

@dataclass
class EvalCfg:
    episodes: int
    deterministic: bool
    render: bool
    video: bool
    model_path: str
    n_envs: int

@dataclass
class Config:
    meta: MetaCfg
    algo: AlgoCfg
    train: TrainCfg
    eval: EvalCfg
    algo_params: Dict[str, Any]