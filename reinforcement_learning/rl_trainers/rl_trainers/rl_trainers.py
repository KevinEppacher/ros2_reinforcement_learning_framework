#!/usr/bin/env python3
import os, sys, glob
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from datetime import datetime
from rl_trainers import BLUE, GREEN, YELLOW, RED, BOLD, RESET
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rl_trainers.configs import Config
from rl_trainers.param_loader import ParamLoader
from rl_trainers.path_manager import PathManager
from rl_trainers.env_manager import EnvManager
from rl_trainers.model_factory import ModelFactory

# ---------------------------- Trainer Node ----------------------------
class RLTrainer(Node):
    def __init__(self):
        super().__init__("rl_trainer")
        # Load config
        self.cfg = ParamLoader(self).load()
        self._log_cfg(self.cfg)
        # Paths
        self.paths = PathManager(self.cfg); self.paths.ensure(); self._log_paths()
        # Envs
        self.envs = EnvManager(self, self.cfg)
        if self.cfg.meta.mode == "train":
            self.train_env = self.envs.make_train()
        else:
            self.eval_env = self.envs.make_eval()
        # Model factory
        self.models = ModelFactory(self, self.cfg, self.paths)

        # Execute
        if self.cfg.meta.mode == "train":
            self._run_train()
        elif self.cfg.meta.mode == "eval":
            self._run_eval()
        else:
            self.get_logger().error(f"Unknown mode '{self.cfg.meta.mode}'"); sys.exit(1)
        self.envs.close()

    # -------------------- operations --------------------
    def _run_train(self):
        from stable_baselines3.common.logger import configure as sb3_configure
        logger = sb3_configure(self.paths.log_dir, ["stdout", "csv", "tensorboard"])
        model = self.models.build(env=self.train_env, for_eval=False)
        model.set_logger(logger)

        # Optional callbacks from spec
        callbacks = []
        if getattr(self.envs.env_build, "make_callbacks", None):
            try:
                callbacks = list(self.envs.env_build.make_callbacks(self))
                self.get_logger().info(f"Using callbacks: {[c.__class__.__name__ for c in callbacks]}")
            except Exception as e:
                self.get_logger().warning(f"make_callbacks failed: {e}")

        model.learn(total_timesteps=int(self.cfg.train.total_timesteps),
                    progress_bar=bool(self.cfg.train.progress),
                    callback=callbacks or None)
        if self.paths.model_path:
            model.save(self.paths.model_path)
        self._maybe_save_vecnorm(self.train_env)

    def _run_eval(self):
        ckpt = self._resolve_checkpoint()
        if not ckpt:
            self.get_logger().error("No checkpoint found for eval."); return
        model = self.models.build(for_eval=True, load_path=ckpt)

        env = getattr(self, "eval_env", None) or self.envs.make_eval()
        env = self._maybe_load_vecnorm(env, ckpt)

        det = bool(self.cfg.eval.deterministic)
        is_vec = isinstance(env, VecEnv)

        if is_vec:
            obs = env.reset()                       # VecEnv: kein (obs, info)
            ep_left = int(self.cfg.eval.episodes)
            rets = np.zeros(env.num_envs, dtype=float)
            while ep_left > 0:
                action, _ = model.predict(obs, deterministic=det)
                obs, rewards, dones, infos = env.step(action)
                rets += np.asarray(rewards, dtype=float)
                for i, d in enumerate(np.atleast_1d(dones)):
                    if d:
                        self.get_logger().info(f"eval env {i} return={rets[i]:.2f}")
                        rets[i] = 0.0
                        ep_left -= 1
                        if ep_left <= 0: break
            env.close()
            return

        # Single Gymnasium env:
        ob = env.reset()
        obs = ob[0] if isinstance(ob, tuple) else ob
        for ep in range(int(self.cfg.eval.episodes)):
            ret, done = 0.0, False
            while not done:
                if isinstance(obs, tuple):  # safety
                    obs = obs[0]
                action, _ = model.predict(obs, deterministic=det)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, r, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, r, done, info = step_out
                ret += float(r)
            self.get_logger().info(f"eval ep {ep+1}/{self.cfg.eval.episodes} return={ret:.2f}")
            ob = env.reset()
            obs = ob[0] if isinstance(ob, tuple) else ob
        env.close()

    # -------------------- utilities --------------------
    def _log_cfg(self, cfg: Config):
        self.get_logger().info(f"{BOLD}{BLUE}===== RLTrainer Parameters ====={RESET}")
        self.get_logger().info(f"[metadata] mode={cfg.meta.mode}, task={cfg.meta.task}, artifacts={cfg.meta.artifacts_dir}")
        self.get_logger().info(f"[algorithm] name={cfg.algo.name}, device={cfg.algo.device}, policy={cfg.algo.policy}")
        self.get_logger().info(f"[train] total_timesteps={cfg.train.total_timesteps}, save_freq={cfg.train.save_freq}, progress={cfg.train.progress}, n_envs={cfg.train.n_envs}")
        self.get_logger().info(f"[eval] episodes={cfg.eval.episodes}, deterministic={cfg.eval.deterministic}, render={cfg.eval.render}, video={cfg.eval.video}, n_envs={cfg.eval.n_envs}")
        if cfg.meta.mode == "train" and cfg.algo_params:
            self.get_logger().info(f"[{cfg.algo.name} params]")
            for k, v in cfg.algo_params.items():
                self.get_logger().info(f"  {k}: {v}")
        self.get_logger().info(f"{BOLD}{BLUE}================================{RESET}")

    def _log_paths(self):
        self.get_logger().info(f"{BOLD}{BLUE}Run timestamp:{RESET} {YELLOW}{datetime.now().strftime('%Y%m%d_%H_%M')}{RESET}")
        self.get_logger().info(f"[paths]:\n  log_dir={YELLOW}{self.paths.log_dir}{RESET}" +
                               (f"\n  model_dir={YELLOW}{self.paths.run_dir}{RESET}\n  model_path={YELLOW}{self.paths.model_path}{RESET}"
                                if self.paths.run_dir else ""))

    def _resolve_checkpoint(self) -> Optional[str]:
        """Prefer explicit eval.model_path, else newest under artifacts/model/<algo>/**."""
        path = (self.cfg.eval.model_path or "").strip()
        if path:
            path = os.path.expanduser(path)
            if not path.endswith(".zip"):
                cand_zip = os.path.join(path, "model.zip")
                cand_plain = os.path.join(path, "model")
                if os.path.isfile(cand_zip): return cand_zip
                if os.path.isfile(cand_plain): return cand_plain
            if os.path.isfile(path): return path
            self.get_logger().error(f"Given eval.model_path does not exist: {path}")
            return None
        root = os.path.join(self.cfg.meta.artifacts_dir, "model", self.cfg.algo.name)
        candidates = glob.glob(os.path.join(root, "**", "*.zip"), recursive=True)
        if not candidates: return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def _unwrap_to(self, env, cls):
        e = env
        try:
            while e is not None:
                if isinstance(e, cls): return e
                e = getattr(e, "venv", None)
        except Exception:
            pass
        return None

    def _maybe_save_vecnorm(self, env):
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            vn = self._unwrap_to(env, VecNormalize)
            if vn is not None and self.cfg.meta.mode == "train":
                path = os.path.join(self.paths.run_dir, "vecnormalize.pkl")
                vn.save(path)
                self.get_logger().info(f"Saved VecNormalize stats: {path}")
        except Exception as e:
            self.get_logger().warning(f"VecNormalize save skipped: {e}")

    def _maybe_load_vecnorm(self, env, ckpt_path):
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            run_dir = os.path.dirname(ckpt_path)
            stats_path = os.path.join(run_dir, "vecnormalize.pkl")
            if os.path.isfile(stats_path):
                env = VecNormalize.load(stats_path, env)
                env.training = False
                env.norm_reward = False
                self.get_logger().info(f"Loaded VecNormalize stats: {stats_path}")
        except Exception as e:
            self.get_logger().warning(f"VecNormalize load skipped: {e}")
        return env

# ---------------------------- main ----------------------------
def main():
    rclpy.init()
    node = RLTrainer()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
