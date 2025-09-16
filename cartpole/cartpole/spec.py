import gymnasium as gym
from rl_core.spec import Spec, EnvBuild
from gymnasium.wrappers import RecordEpisodeStatistics

class CartpoleSpec(Spec):
    def build(self, node=None) -> EnvBuild:

        def make_single(render_mode=None):
            e = gym.make("CartPole-v1", render_mode=render_mode)
            return e

        return EnvBuild(
            make_train_env=lambda: make_single(None),
            make_eval_env =lambda: make_single("human"),
        )

class CartpoleVecSpec(Spec):
    def build(self, node=None) -> EnvBuild:
        def make_single(render_mode=None):
            e = gym.make("CartPole-v1", render_mode=render_mode)
            e = RecordEpisodeStatistics(e)
            # optional: custom reward shaping wrapper hier einhängen
            return e

        return EnvBuild(
            make_train_env=lambda: make_single(None),
            make_eval_env =lambda: make_single("human"),
        )