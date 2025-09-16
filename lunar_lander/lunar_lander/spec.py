#!/usr/bin/env python3
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from rl_core.spec import Spec, EnvBuild
from rl_core.spec import EnvBuild
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from lunar_lander.utils import LanderObsPublisher, StableBaselinesVecPublisher, EpisodePrinter

class LunarLanderSpec(Spec):
    def build(self, node=None) -> EnvBuild:
        if node:
            self.node = node

        def make_single(render_mode=None):
            e = gym.make("LunarLander-v2", render_mode=render_mode)
            return e

        return EnvBuild(
            make_train_env=lambda: make_single(None),
            make_eval_env =lambda: make_single("human"),
        )

class LunarLanderVecSpec:
    def build(self, node=None) -> EnvBuild:
        def make_single(render_mode=None):
            e = gym.make("LunarLander-v2", render_mode=render_mode)
            e = RecordEpisodeStatistics(e)
            e = TimeLimit(e, max_episode_steps=1000)
            return e

        def make_train_vecenv(n_envs: int):
            v = make_vec_env(lambda: make_single(None),
                             n_envs=n_envs,
                             vec_env_cls=SubprocVecEnv,
                             vec_env_kwargs={"start_method": "forkserver"})
            # State/return normalization (stored/loaded by RLTrainer)
            v = VecNormalize(v, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
            return v

        def make_eval_vecenv():
            v = make_vec_env(lambda: make_single("human"),
                             n_envs=1,
                             vec_env_cls=SubprocVecEnv,
                             vec_env_kwargs={"start_method": "forkserver"})
            return v

        def make_callbacks(ros_node):
            # Optional: only if you provided publishers/callbacks
            try:
                pub = LanderObsPublisher(topic="obs_grid", frame_id="lander")
                return [StableBaselinesVecPublisher(pub, every=50),
                        EpisodePrinter(print_env="auto", print_actions=True)]
            except Exception:
                # Fallback: simple episode logger only (if utils are not available yet)
                try:
                    return [EpisodePrinter(print_env="auto", print_actions=True)]
                except Exception:
                    return []

        return EnvBuild(
            make_train_env=lambda: make_single(None),
            make_eval_env=lambda: make_single("human"),
            make_train_vecenv=make_train_vecenv,
            make_eval_vecenv=make_eval_vecenv,
            make_callbacks=make_callbacks,
        )
