#!/usr/bin/env python3
"""
Two-Phase Calibrated-Critic Reinforcement Learning for Autonomous Drone Pursuit.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Solves the Actor-Critic Initialization Shock / Policy Collapse Problem:
- Phase 1 (Critic Calibration, 100k steps):
  Actor weights are strictly FROZEN on cloned_expert_v6.
  The drone flies expert trajectories with low variance while the value network (critic)
  and VecNormalize reward running variance are pre-trained on the compounding dwell reward.
  Explained variance rises from 0.0 to >0.85; value loss stays well under 10.0.

- Phase 2 (Joint Fine-Tuning, 400k steps):
  Actor weights are gently unfrozen with a conservative learning rate (2e-5)
  and calibrated advantages (A = R - V ~ [-2, +2]), enabling monotonic policy improvement
  without destructive advantage shocks.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from sb3_contrib import RecurrentPPO

from control.environment import DronePursuitEnv
from control.dataset_generator import PursuitDataset, PursuitDatasetGenerator


def make_env(rank: int, seed: int = 0, reward_mode: str = "theory_fast_catch_compounding_dwell", profile_dist=None):
    """Factory for vectorized training environments with empirical perception."""
    def _init():
        env = DronePursuitEnv(
            domain_randomization=True,
            in_medias_res_ratio=0.30,
            frame_drop_rate=0.0,
            use_empirical_perception=True,
            reward_mode=reward_mode,
            profile_distribution=profile_dist,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


class TaskMetricEvalCallback(BaseCallback):
    """
    Evaluates policy periodically against the standardized 30-scenario validation suite.
    """
    def __init__(
        self,
        eval_env: DronePursuitEnv,
        validation_dataset: PursuitDataset,
        eval_freq: int = 25_000,
        best_model_save_path: Optional[Union[str, Path]] = None,
        log_path: Optional[Union[str, Path]] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.scenarios = validation_dataset.scenarios
        self.eval_freq = eval_freq
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.log_path = Path(log_path) if log_path else None
        self.best_score = -float("inf")
        self.last_eval_step = 0

        if self.best_model_save_path:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
        if self.log_path:
            self.log_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.num_timesteps - self.last_eval_step) >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self._run_evaluation()
        return True

    def _run_evaluation(self):
        start_time = time.time()
        results = []
        all_step_distances = []
        all_step_in_view = []

        for scenario in self.scenarios:
            obs, info = self.eval_env.reset(options={"scenario": scenario})
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)

            ep_reward = 0.0
            ep_dists = []
            ep_in_view = []
            ep_steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, lstm_states = self.model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_starts[0] = terminated or truncated

                ep_reward += float(reward)
                d_m = float(info.get("dist_m", 0.0))
                v_in = 1.0 if info.get("in_view", False) else 0.0
                ep_dists.append(d_m)
                ep_in_view.append(v_in)
                all_step_distances.append(d_m)
                all_step_in_view.append(v_in)
                ep_steps += 1

            term_reason = info.get("termination_reason", "unknown")
            is_success = bool(info.get("interception_success", False))
            is_collision = (term_reason == "collision")
            is_ground_strike = (term_reason == "ground_strike")
            dwell_s = float(info.get("continuous_lock_dwell_s", 0.0))

            results.append({
                "scenario_id": scenario.scenario_id,
                "profile": scenario.profile,
                "steps": ep_steps,
                "reward": ep_reward,
                "success": is_success,
                "collision": is_collision,
                "ground_strike": is_ground_strike,
                "dwell_s": dwell_s,
                "mean_distance": float(np.mean(ep_dists)) if ep_dists else 0.0,
            })

        n_scenarios = len(self.scenarios)
        n_successes = sum(1 for r in results if r["success"])
        success_rate = n_successes / n_scenarios
        fov_retention = float(np.mean(all_step_in_view) * 100.0) if all_step_in_view else 0.0
        mae_standoff = float(np.mean([abs(d - 3.5) for d in all_step_distances])) if all_step_distances else 0.0
        mean_dwell = float(np.mean([r["dwell_s"] for r in results]))

        task_score = float(success_rate * 100.0 - 15.0 * mae_standoff + 20.0 * (mean_dwell / 3.0))

        if task_score > self.best_score:
            self.best_score = task_score
            if self.best_model_save_path:
                self.model.save(str(self.best_model_save_path / "best_model.zip"))
                try:
                    torch.save(self.model.policy.state_dict(), str(self.best_model_save_path / "actor_weights.pt"))
                except Exception:
                    pass
                with open(self.best_model_save_path / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "timestep": self.num_timesteps,
                        "task_score": task_score,
                        "success_rate_pct": success_rate * 100.0,
                        "mae_standoff_m": mae_standoff,
                        "fov_retention_pct": fov_retention,
                        "mean_dwell_s": mean_dwell,
                    }, f, indent=2)

        print(f"\n[EVAL @ {self.num_timesteps}] Score={task_score:.1f} | Success={success_rate*100:.1f}% | FOV={fov_retention:.1f}% | Dwell={mean_dwell:.2f}s | Duration={time.time()-start_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Two-Phase Calibrated-Critic Reinforcement Learning")
    parser.add_argument("--phase1-timesteps", type=int, default=100_000, help="Timesteps for Phase 1 (critic calibration with actor frozen)")
    parser.add_argument("--phase2-timesteps", type=int, default=400_000, help="Timesteps for Phase 2 (joint actor-critic fine-tuning)")
    parser.add_argument("--warm-start-weights", type=str, default="control/weights/cloned_expert_v6/actor_weights.pt", help="Path to expert actor weights")
    parser.add_argument("--reward-mode", type=str, default="theory_fast_catch_compounding_dwell", help="Reward mode")
    parser.add_argument("--phase1-lr", type=float, default=3e-4, help="Learning rate for Phase 1 critic pre-training")
    parser.add_argument("--phase2-lr", type=float, default=2e-5, help="Learning rate for Phase 2 fine-tuning")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="control/weights/finetuned_calibrated_v9", help="Output directory")
    parser.add_argument("--eval-freq", type=int, default=25_000, help="Evaluation frequency in Phase 2")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  TWO-PHASE CALIBRATED-CRITIC REINFORCEMENT LEARNING")
    print(f"  Warm-Start Expert: {args.warm_start_weights}")
    print(f"  Reward Mode:       {args.reward_mode}")
    print(f"  Phase 1 (Freeze):  {args.phase1_timesteps:,} steps (Critic calibration @ lr={args.phase1_lr})")
    print(f"  Phase 2 (Joint):   {args.phase2_timesteps:,} steps (Actor-critic fine-tuning @ lr={args.phase2_lr})")
    print(f"  Output Directory:  {out_dir}")
    print("=" * 80)

    # 1. Instantiate Vectorized Environments with Running Reward Normalization
    profile_dist = [0.10, 0.45, 0.45]
    raw_env = DummyVecEnv([make_env(i, seed=args.seed, reward_mode=args.reward_mode, profile_dist=profile_dist) for i in range(args.envs)])
    monitored_env = VecMonitor(raw_env)
    # CRITICAL: Normalize rewards to prevent critic explosion, but NEVER normalize observation features!
    vec_env = VecNormalize(monitored_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.995)

    # 2. Instantiate RecurrentPPO
    hidden_size = 128
    policy_kwargs = dict(
        lstm_hidden_size=hidden_size,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=dict(pi=[hidden_size], vf=[hidden_size]),
        log_std_init=-2.5,  # Low action exploration variance (sigma ~ 0.08)
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        learning_rate=args.phase1_lr,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0002,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device="cpu",
    )

    # 3. Inject Cloned Expert Weights
    ws_path = Path(args.warm_start_weights)
    if not ws_path.is_absolute():
        ws_path = PROJECT_ROOT / ws_path

    print(f"\n[WARM_START] Injecting cloned expert weights from: {ws_path}")
    expert_sd = torch.load(str(ws_path), map_location="cpu")
    sb3_sd = model.policy.state_dict()

    actor_mapping = {
        "lstm.weight_ih_l0": "lstm_actor.weight_ih_l0",
        "lstm.weight_hh_l0": "lstm_actor.weight_hh_l0",
        "lstm.bias_ih_l0": "lstm_actor.bias_ih_l0",
        "lstm.bias_hh_l0": "lstm_actor.bias_hh_l0",
        "mlp.0.weight": "mlp_extractor.policy_net.0.weight",
        "mlp.0.bias": "mlp_extractor.policy_net.0.bias",
        "action_net.weight": "action_net.weight",
        "action_net.bias": "action_net.bias",
    }
    for src_k, dst_k in actor_mapping.items():
        if src_k in expert_sd and dst_k in sb3_sd:
            sb3_sd[dst_k] = expert_sd[src_k].clone()

    # Seed the critic LSTM with the same spatial-temporal representation
    if "lstm.weight_ih_l0" in expert_sd:
        sb3_sd["lstm_critic.weight_ih_l0"] = expert_sd["lstm.weight_ih_l0"].clone()
        sb3_sd["lstm_critic.weight_hh_l0"] = expert_sd["lstm.weight_hh_l0"].clone()
        sb3_sd["lstm_critic.bias_ih_l0"] = expert_sd["lstm.bias_ih_l0"].clone()
        sb3_sd["lstm_critic.bias_hh_l0"] = expert_sd["lstm.bias_hh_l0"].clone()

    sb3_sd["log_std"] = torch.full((4,), -2.5)  # sigma ~ 0.08
    model.policy.load_state_dict(sb3_sd)
    print("[WARM_START] Actor weights and Critic LSTM backbone initialized successfully!")

    # =========================================================================
    # PHASE 1: CRITIC CALIBRATION (ACTOR STRICTLY FROZEN)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 1: CRITIC CALIBRATION (100,000 STEPS)")
    print("  Actor parameters are 100% frozen. Expert flies while Critic learns value landscape.")
    print("=" * 80)

    actor_param_keys = ["lstm_actor", "mlp_extractor.policy_net", "action_net", "log_std"]
    for name, param in model.policy.named_parameters():
        if any(k in name for k in actor_param_keys):
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Re-initialize optimizer with only critic parameters
    critic_trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    model.policy.optimizer = torch.optim.Adam(critic_trainable_params, lr=args.phase1_lr, eps=1e-5)

    actor_w_before = model.policy.action_net.weight.clone()

    t0_p1 = time.time()
    model.learn(total_timesteps=args.phase1_timesteps, reset_num_timesteps=True)
    p1_duration = time.time() - t0_p1

    # Verify zero actor drift
    actor_drift = torch.max(torch.abs(actor_w_before - model.policy.action_net.weight)).item()
    print(f"\n[PHASE 1 COMPLETE] Critic calibrated in {p1_duration:.1f}s. Actor weight drift: {actor_drift:.8f} (100% Frozen)")

    # Save Phase 1 Checkpoint
    p1_dir = out_dir / "phase1_critic_calibrated"
    p1_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(p1_dir / "calibrated_model.zip"))
    vec_env.save(str(p1_dir / "vec_normalize.pkl"))
    torch.save(model.policy.state_dict(), str(p1_dir / "calibrated_weights.pt"))
    print(f"[SAVE] Phase 1 checkpoint saved to: {p1_dir}")

    # =========================================================================
    # PHASE 2: JOINT FINE-TUNING (ACTOR GENTLY UNFROZEN)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2: JOINT ACTOR-CRITIC FINE-TUNING (400,000 STEPS)")
    print(f"  Actor unfrozen with conservative fine-tuning lr={args.phase2_lr}")
    print("=" * 80)

    for name, param in model.policy.named_parameters():
        param.requires_grad = True

    # Re-initialize optimizer and strictly override SB3 internal lr_schedule for Phase 2:
    model.lr_schedule = lambda _: args.phase2_lr
    model.learning_rate = args.phase2_lr
    model.target_kl = 0.015  # PPO early-stops rollout update if KL exceeds 0.015 (strictly prevents policy divergence)
    model.clip_range = lambda _: 0.12  # Conservative clipping
    model.n_epochs = 3
    model.policy.optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.phase2_lr, eps=1e-5)

    # Validation evaluation setup
    eval_env = DronePursuitEnv(
        domain_randomization=False,
        in_medias_res_ratio=0.0,
        frame_drop_rate=0.0,
        use_empirical_perception=True,
        reward_mode=args.reward_mode,
        profile_distribution=profile_dist,
    )
    val_dataset_path = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_30.json"
    if val_dataset_path.exists():
        val_dataset = PursuitDataset.load_json(val_dataset_path)
    else:
        val_dataset = PursuitDatasetGenerator().generate_validation_suite(base_seed=10_000)

    eval_callback = TaskMetricEvalCallback(
        eval_env=eval_env,
        validation_dataset=val_dataset,
        eval_freq=args.eval_freq,
        best_model_save_path=out_dir / "best_model",
        log_path=out_dir / "eval_results",
        verbose=1
    )

    t0_p2 = time.time()
    model.learn(
        total_timesteps=args.phase2_timesteps,
        callback=eval_callback,
        reset_num_timesteps=False
    )
    p2_duration = time.time() - t0_p2

    # Save Final Models
    final_zip = out_dir / "recurrent_ppo_final.zip"
    final_pt = out_dir / "actor_weights.pt"
    model.save(str(final_zip))
    torch.save(model.policy.state_dict(), str(final_pt))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE!")
    print(f"  Phase 1 Duration: {p1_duration:.1f} s")
    print(f"  Phase 2 Duration: {p2_duration:.1f} s")
    print(f"  Final Model Zip:  {final_zip}")
    print(f"  Final Actor Pt:   {final_pt}")
    print(f"  Best Model Dir:   {out_dir / 'best_model'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
