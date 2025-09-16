# car_racing/spec.py
import gymnasium as gym
from rl_core.spec import EnvBuild
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit, ClipAction
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CallbackList
from car_racing.wrappers import DiscretizeAction, ToGray, CropBottom, ClipReward, OfftrackTimeout
from car_racing.utils import CarRacingObsPublisher, StableBaselinesVecPublisher, EpisodePrinter

class CarRacingSpec:
    def build(self, node=None) -> EnvBuild:
        def make_single(render_mode=None):
            e = gym.make("CarRacing-v2", render_mode=render_mode, domain_randomize=False)
            e = ClipAction(e)
            e = DiscretizeAction(e)
            e = ToGray(e)
            e = CropBottom(e, crop_pixels=12)
            e = ClipReward(e, rmin=-1.0, rmax=1.0)
            e = OfftrackTimeout(e, max_off_steps=200, offroad_threshold=1e-3)
            e = RecordEpisodeStatistics(e)
            e = TimeLimit(e, max_episode_steps=500)
            return e

        def make_train_vecenv(n_envs: int):
            v = make_vec_env(lambda: make_single(None), n_envs=n_envs,
                             vec_env_cls=SubprocVecEnv,
                             vec_env_kwargs={"start_method": "forkserver"})
            v = VecTransposeImage(v)
            v = VecFrameStack(v, n_stack=4, channels_order="first")
            v = VecNormalize(v, norm_obs=False, norm_reward=True, clip_obs=10.0, gamma=0.99)
            return v

        def make_eval_vecenv():
            v = make_vec_env(lambda: make_single("human"), n_envs=1,
                             vec_env_cls=SubprocVecEnv,
                             vec_env_kwargs={"start_method": "forkserver"})
            v = VecTransposeImage(v)
            v = VecFrameStack(v, n_stack=4, channels_order="first")
            return v

        def make_callbacks(ros_node):
            # build ROS pub + callbacks here if gewünscht
            pub = CarRacingObsPublisher(topic="obs_grid", frame_id="camera")
            return [StableBaselinesVecPublisher(pub, every=25), EpisodePrinter("auto", True)]

        return EnvBuild(
            make_train_env=lambda: make_single(None),
            make_eval_env=lambda: make_single("human"),
            make_train_vecenv=make_train_vecenv,
            make_eval_vecenv=make_eval_vecenv,
            make_callbacks=make_callbacks,
        )
