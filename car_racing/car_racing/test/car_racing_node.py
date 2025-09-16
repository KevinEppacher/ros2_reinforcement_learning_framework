#!/usr/bin/env python3
# coding: utf-8
"""
CarRacing-v2 PPO training & evaluation (modular, configurable).
- Configure observation (grayscale/resize, frame stack)
- Configure action space (continuous or discretized)
- Configure reward shaping (scale, step penalty)
- Tracks wall-clock training time and SPS
- Logs to CSV + TensorBoard and saves a loss diagram (loss_curve.png)
- Saves/loads models under: <pkg>/data/{logs,models,videos}
"""

import argparse, os, time, pathlib, datetime as dt
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from PIL import Image

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecTransposeImage, VecFrameStack, VecVideoRecorder, VecNormalize
)
from stable_baselines3.common.logger import configure as sb3_configure_logger

# ---------- Project paths (kept from your original code) ----------
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
LOGS_DIR  = DATA_ROOT / "logs"
MODELS_DIR= DATA_ROOT / "models"
VIDEOS_DIR= DATA_ROOT / "videos"
for d in [LOGS_DIR, MODELS_DIR, VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- Wrappers (obs/action/reward) ----------

class ToGray(gym.ObservationWrapper):
    """Convert RGB (H,W,3) to grayscale (H,W,1)."""
    def __init__(self, env, keep_dim=True):
        super().__init__(env)
        self.keep_dim = keep_dim
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, 1 if keep_dim else ()), dtype=np.uint8
        )
    def observation(self, obs):
        gray = (0.299*obs[...,0] + 0.587*obs[...,1] + 0.114*obs[...,2]).astype(np.uint8)
        return gray[..., None] if self.keep_dim else gray

class ResizeObs(gym.ObservationWrapper):
    """Resize to (size, size) using bilinear resample."""
    def __init__(self, env, size: int):
        super().__init__(env)
        self.size = int(size)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.size, self.size, c), dtype=np.uint8
        )
    def observation(self, obs):
        arr = obs if obs.ndim == 3 and (obs.shape[2] != 1) else obs.squeeze(-1)
        im = Image.fromarray(arr)
        im = im.resize((self.size, self.size), Image.BILINEAR)
        out = np.asarray(im)
        if out.ndim == 2:
            out = out[..., None]
        return out

class ActionRepeat(gym.Wrapper):
    """Repeat each action 'repeat' steps to reduce sim overhead."""
    def __init__(self, env, repeat=1):
        super().__init__(env)
        self.repeat = int(max(1, repeat))
    def step(self, action):
        total_r = 0.0
        terminated = truncated = False
        info = {}
        obs = None
        for _ in range(self.repeat):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_r += float(r)
            if terminated or truncated:
                break
        return obs, total_r, terminated, truncated, info

class DiscretizeAction(gym.ActionWrapper):
    """Map a discrete index -> continuous [steer, throttle, brake]."""
    def __init__(self, env, steer_bins=5, throttle_bins=3, brake_bins=2):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box) and env.action_space.shape == (3,)
        self.steers    = np.linspace(-1.0, 1.0, steer_bins)
        self.throttles = np.linspace(0.0, 1.0, throttle_bins)
        self.brakes    = np.linspace(0.0, 1.0, brake_bins)
        self._lut = [np.array([s, t, b], dtype=np.float32)
                    for s in self.steers for t in self.throttles for b in self.brakes]
        self.action_space = gym.spaces.Discrete(len(self._lut))
    def action(self, a):
        return self._lut[int(a)]

class RewardShaping(gym.Wrapper):
    """Linear shaping: scaled reward and per-step penalty."""
    def __init__(self, env, scale=1.0, step_penalty=0.0):
        super().__init__(env)
        self.scale = float(scale); self.step_penalty = float(step_penalty)
    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        r = self.scale * float(r) - self.step_penalty
        return obs, r, terminated, truncated, info

# ---------- Config dataclasses ----------

@dataclass
class ObsConfig:
    grayscale: bool = False
    resize: int | None = None     # e.g., 84
    frame_stack: int = 4

@dataclass
class ActConfig:
    discretize: bool = False
    steer_bins: int = 5
    throttle_bins: int = 3
    brake_bins: int = 2
    action_repeat: int = 1

@dataclass
class RewardConfig:
    scale: float = 1.0
    step_penalty: float = 0.0
    normalize_reward: bool = False  # VecNormalize on rewards only

# ---------- Env builders ----------

def make_single_env(env_id: str, render_mode: str | None, seed: int,
                    obs_cfg: ObsConfig, act_cfg: ActConfig, rew_cfg: RewardConfig):
    """Build a single CarRacing env with requested wrappers."""
    e = gym.make(env_id, render_mode=render_mode)
    if rew_cfg.scale != 1.0 or rew_cfg.step_penalty != 0.0:
        e = RewardShaping(e, scale=rew_cfg.scale, step_penalty=rew_cfg.step_penalty)
    if act_cfg.action_repeat > 1:
        e = ActionRepeat(e, repeat=act_cfg.action_repeat)
    if act_cfg.discretize:
        e = DiscretizeAction(e, act_cfg.steer_bins, act_cfg.throttle_bins, act_cfg.brake_bins)
    if obs_cfg.grayscale:
        e = ToGray(e, keep_dim=True)
    if obs_cfg.resize:
        e = ResizeObs(e, obs_cfg.resize)
    e = RecordEpisodeStatistics(e)
    e.reset(seed=seed)
    return e

def make_vec(env_id: str, n_envs: int, seed: int, is_eval: bool,
             obs_cfg: ObsConfig, act_cfg: ActConfig, rew_cfg: RewardConfig,
             video_dir: Path | None = None, video_len: int = 1000):
    """Vectorized env with proper channel order and frame stacking."""
    render_mode = "rgb_array" if is_eval else None
    env = make_vec_env(
        lambda: make_single_env(env_id, render_mode, seed, obs_cfg, act_cfg, rew_cfg),
        n_envs=n_envs, seed=seed
    )
    env = VecTransposeImage(env)  # (H,W,C)->(C,H,W)
    env = VecFrameStack(env, n_stack=obs_cfg.frame_stack, channels_order="first")

    if rew_cfg.normalize_reward and not is_eval:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if is_eval and video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)
        env = VecVideoRecorder(
            env, video_folder=str(video_dir),
            record_video_trigger=lambda step: step == 0,
            video_length=video_len, name_prefix="ppo_carracing",
        )
    return env

# ---------- Logging and plotting ----------

def prepare_logger(base_log_dir: Path):
    """Create unique run dir and configure SB3 logger for stdout+csv+tensorboard."""
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_log_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = sb3_configure_logger(str(run_dir), ["stdout", "csv", "tensorboard"])
    return logger, run_dir

def plot_losses(run_dir: Path, out_name: str = "loss_curve.png"):
    """Read SB3 CSV (progress.csv) and save a simple loss diagram."""
    csv_path = run_dir / "progress.csv"
    if not csv_path.exists():
        print(f"[warn] CSV not found: {csv_path}")
        return
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        keys = [k for k in [
            "train/value_loss", "train/policy_gradient_loss",
            "train/entropy_loss", "train/approx_kl", "time/total_timesteps"
        ] if k in df.columns]
        if not keys:
            print("[warn] No loss columns found to plot.")
            return
        x = df["time/total_timesteps"] if "time/total_timesteps" in df.columns else np.arange(len(df))
        plt.figure(figsize=(8,4.5))
        for k in keys:
            if k == "time/total_timesteps": continue
            plt.plot(x, df[k], label=k)
        plt.xlabel("total timesteps"); plt.ylabel("loss / metric")
        plt.title("PPO training losses"); plt.legend()
        out_file = run_dir / out_name
        plt.tight_layout(); plt.savefig(out_file, dpi=150)
        print(f"[info] Saved loss diagram -> {out_file}")
    except Exception as e:
        print(f"[warn] Could not plot losses: {e}")

# ---------- Train / Eval ----------

def train(args):
    # thread caps help vectorized env scaling
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass

    obs_cfg = ObsConfig(grayscale=args.grayscale, resize=args.resize, frame_stack=args.frame_stack)
    act_cfg = ActConfig(
        discretize=args.discretize_actions,
        steer_bins=args.steer_bins, throttle_bins=args.throttle_bins,
        brake_bins=args.brake_bins, action_repeat=args.action_repeat
    )
    rew_cfg = RewardConfig(scale=args.reward_scale, step_penalty=args.step_penalty,
                           normalize_reward=args.normalize_reward)

    env = make_vec(args.env_id, args.n_envs, args.seed, False, obs_cfg, act_cfg, rew_cfg)
    model = PPO(
        "CnnPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        tensorboard_log=str(LOGS_DIR),
        verbose=1
    )

    logger, run_dir = prepare_logger(LOGS_DIR)
    model.set_logger(logger)
    print(f"[info] Logging to: {run_dir}")

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, progress_bar=args.progress)
    dt_sec = time.perf_counter() - t0
    sps = args.timesteps / max(dt_sec, 1e-9)
    print(f"[info] Train wall-time: {dt_sec:.1f}s  |  SPS≈{sps:.0f}")

    model_path = MODELS_DIR / f"ppo_carracing_{args.timesteps}"
    model.save(model_path)
    print(f"[info] Model saved to {model_path}.zip")

    # save VecNormalize statistics when used
    try:
        from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VN
        if isinstance(env, VN):
            stats_path = model_path.with_suffix(".vecnorm.pkl")
            env.save(str(stats_path))
            print(f"[info] Saved VecNormalize stats -> {stats_path}")
    except Exception:
        pass

    plot_losses(run_dir, out_name="loss_curve.png")
    env.close()

def evaluate(args):
    obs_cfg = ObsConfig(grayscale=args.grayscale, resize=args.resize, frame_stack=args.frame_stack)
    act_cfg = ActConfig(
        discretize=args.discretize_actions,
        steer_bins=args.steer_bins, throttle_bins=args.throttle_bins,
        brake_bins=args.brake_bins, action_repeat=args.action_repeat
    )
    rew_cfg = RewardConfig(scale=args.reward_scale, step_penalty=args.step_penalty,
                           normalize_reward=False)  # never normalize reward on eval rollout

    # resolve model path under MODELS_DIR if relative
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = MODELS_DIR / model_path
    assert model_path.with_suffix(".zip").exists(), f"Model not found: {model_path}.zip"

    eval_env = make_vec(args.env_id, 1, args.seed, True, obs_cfg, act_cfg, rew_cfg,
                        video_dir=VIDEOS_DIR, video_len=args.video_len)

    # load model & vecnorm stats if present
    model = PPO.load(model_path, device="auto")
    vn_stats = model_path.with_suffix(".vecnorm.pkl")
    if vn_stats.exists():
        from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
        eval_env = VecNormalize.load(str(vn_stats), eval_env)
        eval_env.training = False; eval_env.norm_reward = False
        print(f"[info] Loaded VecNormalize stats from {vn_stats}")

    obs = eval_env.reset()
    total_r = 0.0
    try:
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, done, info = eval_env.step(action)
            total_r += float(r[0])
            if done[0]:
                break
    finally:
        eval_env.close()
    print(f"[info] Episode Reward: {total_r:.2f}")

# ---------- CLI ----------

def build_argparser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared
    p.add_argument("--env-id", type=str, default="CarRacing-v2")

    # train
    pt = sub.add_parser("train")
    pt.add_argument("--timesteps", type=int, default=300_000)
    pt.add_argument("--n-envs", type=int, default=8)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--n-steps", type=int, default=2048)
    pt.add_argument("--batch-size", type=int, default=256)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--ent-coef", type=float, default=0.01)
    pt.add_argument("--vf-coef", type=float, default=0.5)
    pt.add_argument("--clip-range", type=float, default=0.2)
    pt.add_argument("--progress", action="store_true", help="show tqdm progress bar")
    pt.add_argument("--normalize-reward", action="store_true")

    # eval
    pe = sub.add_parser("eval")
    pe.add_argument("--model", type=str, required=True)  # e.g., ppo_carracing_300000
    pe.add_argument("--steps", type=int, default=1500)
    pe.add_argument("--seed", type=int, default=123)
    pe.add_argument("--video-len", type=int, default=1000)
    pe.add_argument("--deterministic", action="store_true")

    # observation config
    for sp in (pt, pe):
        sp.add_argument("--grayscale", action="store_true")
        sp.add_argument("--resize", type=int, default=None, help="square size, e.g. 84; omit for native 96")
        sp.add_argument("--frame-stack", type=int, default=4)

    # action config
    for sp in (pt, pe):
        sp.add_argument("--discretize-actions", action="store_true")
        sp.add_argument("--steer-bins", type=int, default=5)
        sp.add_argument("--throttle-bins", type=int, default=3)
        sp.add_argument("--brake-bins", type=int, default=2)
        sp.add_argument("--action-repeat", type=int, default=1)

    # reward config
    for sp in (pt, pe):
        sp.add_argument("--reward-scale", type=float, default=1.0)
        sp.add_argument("--step-penalty", type=float, default=0.0)

    return p

def main():
    args = build_argparser().parse_args()
    if args.cmd == "train":
        train(args)
    else:
        evaluate(args)
    print("Done")

if __name__ == "__main__":
    main()
