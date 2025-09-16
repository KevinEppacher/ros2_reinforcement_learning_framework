import gymnasium as gym
from stable_baselines3 import PPO
from datetime import datetime
import os
import sys

# Zeitstempel nur für Log-Verzeichnis (optional)
now = datetime.now()
date = now.strftime("%Y%m%d")
hour = now.strftime("%H")
minute = now.strftime("%M")
log_dir = f"/app/src/car_racing/data/logs/PPO/eval_car_racing_{date}_{hour}_{minute}"
os.makedirs(log_dir, exist_ok=True)

# Pfad BASIS (ohne .zip) des bestehenden Modells anpassen:
# MODEL_BASE = "/app/src/car_racing/data/model/PPO/car_racing_20250909_10_23/model"
MODEL_BASE = "/app/notebook/models/ppo_carracing_30000"

model_zip = MODEL_BASE + ".zip"
if not os.path.isfile(model_zip):
    print(f"[ERROR] Modell fehlt: {model_zip}")
    sys.exit(1)

print(f"[INFO] Lade Modell: {model_zip}")

# Nur Evaluation (render_mode="human" für Fenster)
env = gym.make("CarRacing-v2", render_mode="human")

model = PPO.load(MODEL_BASE)  # kein env nötig zum reinen Predict

EPISODES = 5
MAX_STEPS = 200  # pro Episode (CarRacing Limit)

for ep in range(1, EPISODES + 1):
    obs, info = env.reset()
    ep_ret = 0.0
    for t in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += float(reward)
        if terminated or truncated:
            break
    print(f"[EVAL] Episode {ep}: return={ep_ret:.2f} steps={t+1}")

env.close()
print("[INFO] Evaluation fertig.")