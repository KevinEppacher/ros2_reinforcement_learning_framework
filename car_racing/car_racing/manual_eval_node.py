#!/usr/bin/env python3
# coding: utf-8
"""
ROS2 node for manual driving and modular reward evaluation in Gymnasium CarRacing.
- Keyboard control (pygame)
- Pluggable reward terms (composite shaper)
- Publishes raw, shaped, and per-term rewards
ros2 run car_racing manual_eval_node --ros-args \
  -p env_id:=CarRacing-v2 -p fps:=50 -p scale:=6.0 \
  -p reward.use_spec:=false \
  -p reward.w.env:=1.0 \
  -p reward.w.brake_pen:=0.2 \
  -p reward.w.steer_smooth:=0.1 \
  -p reward.w.throttle_smooth:=0.05 \
  -p reward.w.speed:=0.0 -p reward.speed.target:=0.012 -p reward.speed.tol:=0.008
"""
import os
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_RENDER_VSYNC", "1")

import time
import numpy as np
import gymnasium as gym
import pygame
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

try:
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    import cv2
except Exception:
    Image = None
    CvBridge = None
    cv2 = None


# -------- Reward shaping framework --------
class RewardTerm:
    """Interface for a single reward component."""
    name: str = "term"
    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)
    def reset(self):  # called on episode reset
        pass
    def compute(self, obs, action: np.ndarray, env_r: float, info: dict, ctx: dict) -> float:
        return 0.0


class EnvRewardTerm(RewardTerm):
    """Pass-through of environment reward."""
    name = "env"
    def compute(self, obs, action, env_r, info, ctx) -> float:
        return float(env_r)


class BrakePenaltyTerm(RewardTerm):
    """Penalize brake usage to encourage smoother driving."""
    name = "brake_pen"
    def compute(self, obs, action, env_r, info, ctx) -> float:
        # action = [steer, throttle, brake]
        return -float(action[2])


class SteeringSmoothnessTerm(RewardTerm):
    """Penalize steering changes to reduce oscillation."""
    name = "steer_smooth"
    def reset(self):
        self._prev = None
    def compute(self, obs, action, env_r, info, ctx) -> float:
        steer = float(action[0])
        prev = ctx.setdefault("prev_steer", None)
        ctx["prev_steer"] = steer
        if prev is None:
            return 0.0
        return -abs(steer - prev)


class ThrottleSmoothnessTerm(RewardTerm):
    """Penalize throttle changes to reduce jerks."""
    name = "throttle_smooth"
    def reset(self):
        self._prev = None
    def compute(self, obs, action, env_r, info, ctx) -> float:
        thr = float(action[1])
        prev = ctx.setdefault("prev_thr", None)
        ctx["prev_thr"] = thr
        if prev is None:
            return 0.0
        return -abs(thr - prev)


class SpeedTerm(RewardTerm):
    """Reward target speed if available in info; otherwise zero."""
    name = "speed"
    def __init__(self, weight: float = 1.0, target: float = 0.012, tol: float = 0.008):
        super().__init__(weight)
        self.target = float(target)
        self.tol = float(tol)
    def compute(self, obs, action, env_r, info, ctx) -> float:
        # Gymnasium CarRacing may expose speed in info, else 0.0
        v = float(info.get("speed", 0.0))
        # simple band reward: max at target, falls off with L1 distance
        return -abs(v - self.target) / max(self.tol, 1e-6)


class OnTrackBonusTerm(RewardTerm):
    """Optional: bonus when on track if info provides flag."""
    name = "on_track"
    def __init__(self, weight: float = 1.0, bonus: float = 1.0):
        super().__init__(weight)
        self.bonus = float(bonus)
    def compute(self, obs, action, env_r, info, ctx) -> float:
        flag = info.get("on_track", None)
        if flag is None:
            # Try inverse of offroad if available
            off = info.get("offroad", None)
            if off is None:
                return 0.0
            return self.bonus * (1.0 - float(off))
        return self.bonus if bool(flag) else 0.0


class SpecAdapterTerm(RewardTerm):
    """Adapts existing task spec.process_reward(obs, action, env_r, info)."""
    name = "spec"
    def __init__(self, spec_obj, weight: float = 1.0):
        super().__init__(weight)
        self.spec = spec_obj
    def compute(self, obs, action, env_r, info, ctx) -> float:
        try:
            return float(self.spec.process_reward(obs, action, env_r, info))
        except Exception:
            return 0.0


class CompositeReward:
    """Aggregates multiple terms: shaped = sum_i w_i * term_i."""
    def __init__(self, terms: List[RewardTerm]):
        self.terms = terms
        self.ctx: dict = {}
    def reset(self):
        self.ctx.clear()
        for t in self.terms:
            t.reset()
    def compute(self, obs, action, env_r, info) -> Tuple[float, Dict[str, float]]:
        vals = {}
        total = 0.0
        for t in self.terms:
            v = t.compute(obs, action, env_r, info, self.ctx)
            v_w = t.weight * v
            vals[t.name] = float(v_w)
            total += v_w
        return float(total), vals


# -------- Optional spec loader --------
def _load_spec():
    try:
        from car_racing.spec import CarRacingSpec
        return CarRacingSpec()
    except Exception:
        return None


# -------- Input mapping --------
def _action_from_keys(keys) -> np.ndarray:
    """Map pressed pygame keys to CarRacing action [steer, throttle, brake]."""
    a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        a[0] = -1.0
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        a[0] = 1.0 if a[0] == 0.0 else a[0]
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        a[1] = 1.0
    if keys[pygame.K_s] or keys[pygame.K_SPACE] or keys[pygame.K_DOWN]:
        a[2] = 1.0
    return a


# -------- ROS2 Node --------
class CarRacingManualEval(Node):
    def __init__(self):
        super().__init__("car_racing_manual_eval")

        # --- parameters ---
        self.env_id = str(self.declare_parameter("env_id", "CarRacing-v2").value)
        self.render_mode = str(self.declare_parameter("render_mode", "rgb_array").value)  # "rgb_array"|"human"
        self.fps = int(self.declare_parameter("fps", 50).value)
        self.seed = int(self.declare_parameter("seed", 0).value)
        self.scale = float(self.declare_parameter("scale", 6.0).value)
        self.publish_image = bool(self.declare_parameter("publish_image", True).value)

        # reward config
        self.enable_spec = bool(self.declare_parameter("reward.use_spec", False).value)
        # weights
        w_env = float(self.declare_parameter("reward.w.env", 1.0).value)
        w_brk = float(self.declare_parameter("reward.w.brake_pen", 0.0).value)
        w_steer = float(self.declare_parameter("reward.w.steer_smooth", 0.1).value)
        w_thr = float(self.declare_parameter("reward.w.throttle_smooth", 0.05).value)
        w_speed = float(self.declare_parameter("reward.w.speed", 0.0).value)
        w_track = float(self.declare_parameter("reward.w.on_track", 0.0).value)
        speed_target = float(self.declare_parameter("reward.speed.target", 0.012).value)
        speed_tol = float(self.declare_parameter("reward.speed.tol", 0.008).value)

        # --- gym env ---
        self.env = gym.make(self.env_id, render_mode="rgb_array")
        self.env.reset(seed=self.seed)
        self.spec = _load_spec() if self.enable_spec else None

        # --- reward shaper ---
        terms: List[RewardTerm] = [EnvRewardTerm(weight=w_env)]
        if self.enable_spec and self.spec is not None and hasattr(self.spec, "process_reward"):
            terms.append(SpecAdapterTerm(self.spec, weight=1.0))  # keep weight 1.0; adjust in spec if needed
        if w_brk != 0.0: terms.append(BrakePenaltyTerm(weight=w_brk))
        if w_steer != 0.0: terms.append(SteeringSmoothnessTerm(weight=w_steer))
        if w_thr != 0.0: terms.append(ThrottleSmoothnessTerm(weight=w_thr))
        if w_speed != 0.0: terms.append(SpeedTerm(weight=w_speed, target=speed_target, tol=speed_tol))
        if w_track != 0.0: terms.append(OnTrackBonusTerm(weight=w_track, bonus=1.0))
        self.rew = CompositeReward(terms)
        self.rew.reset()

        # --- ROS pubs ---
        self.pub_raw = self.create_publisher(Float32, "reward/raw", 10)
        self.pub_shaped = self.create_publisher(Float32, "reward/shaped", 10)
        self.term_pubs: Dict[str, any] = {}
        for t in terms:
            self.term_pubs[t.name] = self.create_publisher(Float32, f"reward/term/{t.name}", 10)

        self.pub_img = None
        self.bridge = None
        if self.publish_image and Image is not None and CvBridge is not None:
            self.pub_img = self.create_publisher(Image, "image", 2)
            self.bridge = CvBridge()

        # --- pygame window ---
        pygame.init()
        obs, _ = self.env.reset()
        h, w = obs.shape[:2]
        self.win_w, self.win_h = int(w * self.scale), int(h * self.scale)
        self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.DOUBLEBUF)
        pygame.display.set_caption("CarRacing Manual Eval")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, max(14, int(18 * self.scale / 4)))

        # episodic stats
        self.ep_ret_raw = 0.0
        self.ep_ret_shaped = 0.0
        self.ep_len = 0
        self.episodes = 0
        self.running = True

    def step_loop(self):
        obs, _ = self.env.reset(seed=self.seed)
        self.rew.reset()
        while rclpy.ok() and self.running:
            # events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    if event.key == pygame.K_r:
                        obs, _ = self.env.reset()
                        self.rew.reset()
                        self.ep_ret_raw = self.ep_ret_shaped = 0.0
                        self.ep_len = 0

            keys = pygame.key.get_pressed()
            action = _action_from_keys(keys)

            obs, r_env, terminated, truncated, info = self.env.step(action)
            r_shaped, term_vals = self.rew.compute(obs, action, r_env, info)

            self.ep_ret_raw += float(r_env)
            self.ep_ret_shaped += float(r_shaped)
            self.ep_len += 1

            # publish rewards
            self.pub_raw.publish(Float32(data=float(r_env)))
            self.pub_shaped.publish(Float32(data=float(r_shaped)))
            for name, val in term_vals.items():
                self.term_pubs[name].publish(Float32(data=float(val)))

            # render
            frame = obs  # 96x96 RGB
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            if self.scale != 1.0:
                surf = pygame.transform.smoothscale(surf, (self.win_w, self.win_h))
            self.screen.blit(surf, (0, 0))

            # HUD
            hud = [
                f"step {self.ep_len}",
                f"raw {self.ep_ret_raw:.2f}   shaped {self.ep_ret_shaped:.2f}",
                " + ".join([f"{k}:{term_vals.get(k,0.0):+.2f}" for k in self.term_pubs.keys()]),
                "A/D steer | W throttle | S/SPACE brake | R reset | Q quit",
            ]
            y = 5
            for line in hud:
                self.screen.blit(self.font.render(line, True, (255, 255, 255)), (5, y))
                y += 16
            pygame.display.flip()

            # optional ROS image
            if self.pub_img and self.bridge:
                img_bgr = frame[:, :, ::-1].copy()
                msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "car_racing_camera"
                self.pub_img.publish(msg)

            if terminated or truncated:
                self.episodes += 1
                self.get_logger().info(
                    f"episode={self.episodes} len={self.ep_len} raw={self.ep_ret_raw:.2f} shaped={self.ep_ret_shaped:.2f}"
                )
                obs, _ = self.env.reset()
                self.rew.reset()
                self.ep_ret_raw = self.ep_ret_shaped = 0.0
                self.ep_len = 0

            rclpy.spin_once(self, timeout_sec=0.0)
            self.clock.tick(self.fps)

        pygame.quit()
        self.env.close()


def main():
    rclpy.init()
    node = CarRacingManualEval()
    try:
        node.step_loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
