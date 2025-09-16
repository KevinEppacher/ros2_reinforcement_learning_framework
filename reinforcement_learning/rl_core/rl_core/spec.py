from dataclasses import dataclass
from typing import Callable, Optional, Any, TYPE_CHECKING
import numpy as np
import gymnasium as gym
# rl_core/spec.py
from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict
import gymnasium as gym
import numpy as np
# Only import for type-checkers (no runtime dependency if SB3 not yet imported)
if TYPE_CHECKING:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

@dataclass
class EnvBuild:
    # Single-env factories (fallback)
    make_train_env: Callable[[], gym.Env]
    make_eval_env: Callable[[], gym.Env]

    # Optional vectorized factories
    make_train_vecenv: Optional[Callable[[int], "VecEnv"]] = None
    make_eval_vecenv: Optional[Callable[[], "VecEnv"]] = None

    # Optional callback factory: returns a list[BaseCallback]
    make_callbacks: Optional[Callable[[Any], list]] = None

    # Optional image converter
    obs_to_mono: Optional[Callable[[np.ndarray], np.ndarray]] = None

class Spec:
    """Plugin interface each env package implements."""
    def build(self, params: Dict[str, Any]) -> EnvBuild:
        raise NotImplementedError
