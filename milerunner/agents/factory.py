"""Build Stable-Baselines3 agents from a hyperparameter/architecture spec.

Supports the full roster required by the brief: PPO, SAC, TD3, DDPG and A2C
(the synchronous form of A3C), plus recurrent PPO (LSTM over time) when
``sb3_contrib`` is available. Each agent's feature extractor is chosen from
:mod:`milerunner.agents.networks`, so the evolutionary layer can mutate both
the RL algorithm and the network architecture.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .networks import make_extractor_kwargs

try:
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise
    _HAVE_SB3 = True
except Exception:  # pragma: no cover
    _HAVE_SB3 = False

try:
    from sb3_contrib import RecurrentPPO
    _HAVE_RECURRENT = True
except Exception:  # pragma: no cover
    _HAVE_RECURRENT = False


ON_POLICY = {"ppo", "a2c", "a3c", "ppo_lstm"}
OFF_POLICY = {"sac", "td3", "ddpg"}
ALL_ALGOS = sorted(ON_POLICY | OFF_POLICY)


def _net_arch(hidden: int, depth: int, algo: str):
    layers = [hidden] * max(depth, 1)
    if algo in ("sac", "td3", "ddpg"):
        return layers               # shared list for off-policy
    return dict(pi=layers, vf=layers)


def build_agent(algo: str, env, hyperparams: Optional[Dict[str, Any]] = None,
                arch: str = "mlp", device: str = "auto",
                tensorboard_log: Optional[str] = None, seed: Optional[int] = None,
                verbose: int = 0):
    """Instantiate an SB3 model for ``algo`` on ``env``.

    ``hyperparams`` may include: learning_rate, gamma, ent_coef, batch_size,
    n_steps, gae_lambda, clip_range, tau, gradient_steps, features_dim,
    hidden, depth, activation, and architecture-specific extras.
    """
    if not _HAVE_SB3:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required to build agents")
    algo = algo.lower()
    hp = dict(hyperparams or {})

    features_dim = int(hp.pop("features_dim", 256))
    hidden = int(hp.pop("hidden", 256))
    depth = int(hp.pop("depth", 2))
    activation = hp.pop("activation", "relu")
    arch_extra = hp.pop("arch_kwargs", {})

    policy_kwargs: Dict[str, Any] = dict(
        net_arch=_net_arch(hidden, depth, algo),
        **make_extractor_kwargs(arch, features_dim, arch_extra),
    )
    if algo not in ("ppo_lstm",):
        policy_kwargs["activation_fn"] = {
            "relu": __import__("torch").nn.ReLU,
            "tanh": __import__("torch").nn.Tanh,
            "gelu": __import__("torch").nn.GELU,
        }.get(activation, __import__("torch").nn.ReLU)

    common = dict(
        policy="MlpPolicy",
        env=env,
        device=device,
        seed=seed,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
    )
    lr = float(hp.get("learning_rate", 3e-4))
    gamma = float(hp.get("gamma", 0.99))

    if algo in ("ppo", "a3c") or algo == "ppo":
        return PPO(
            learning_rate=lr, gamma=gamma,
            n_steps=int(hp.get("n_steps", 2048)),
            batch_size=int(hp.get("batch_size", 256)),
            gae_lambda=float(hp.get("gae_lambda", 0.95)),
            clip_range=float(hp.get("clip_range", 0.2)),
            ent_coef=float(hp.get("ent_coef", 0.0)),
            n_epochs=int(hp.get("n_epochs", 10)),
            **common,
        )
    if algo == "a2c":
        return A2C(
            learning_rate=lr, gamma=gamma,
            n_steps=int(hp.get("n_steps", 8)),
            gae_lambda=float(hp.get("gae_lambda", 1.0)),
            ent_coef=float(hp.get("ent_coef", 0.0)),
            **common,
        )
    if algo == "ppo_lstm":
        if not _HAVE_RECURRENT:  # pragma: no cover
            raise RuntimeError("sb3_contrib required for ppo_lstm")
        common["policy"] = "MlpLstmPolicy"
        # recurrent policy manages its own extractor; drop custom one
        common["policy_kwargs"] = dict(net_arch=_net_arch(hidden, depth, "ppo"),
                                       lstm_hidden_size=int(hp.get("lstm_hidden", 128)))
        return RecurrentPPO(
            learning_rate=lr, gamma=gamma,
            n_steps=int(hp.get("n_steps", 512)),
            batch_size=int(hp.get("batch_size", 128)),
            gae_lambda=float(hp.get("gae_lambda", 0.95)),
            ent_coef=float(hp.get("ent_coef", 0.0)),
            **common,
        )

    # ---- off-policy ----
    n_actions = env.action_space.shape[-1] if hasattr(env, "action_space") else \
        env.get_attr("action_space")[0].shape[-1]
    if algo == "sac":
        return SAC(
            learning_rate=lr, gamma=gamma,
            batch_size=int(hp.get("batch_size", 256)),
            tau=float(hp.get("tau", 0.005)),
            train_freq=int(hp.get("train_freq", 1)),
            gradient_steps=int(hp.get("gradient_steps", 1)),
            buffer_size=int(hp.get("buffer_size", 300_000)),
            learning_starts=int(hp.get("learning_starts", 1000)),
            ent_coef=hp.get("ent_coef", "auto"),
            **common,
        )
    noise = NormalActionNoise(mean=np.zeros(n_actions),
                              sigma=float(hp.get("action_noise", 0.1)) * np.ones(n_actions))
    if algo == "td3":
        return TD3(
            learning_rate=lr, gamma=gamma,
            batch_size=int(hp.get("batch_size", 256)),
            tau=float(hp.get("tau", 0.005)),
            buffer_size=int(hp.get("buffer_size", 300_000)),
            learning_starts=int(hp.get("learning_starts", 1000)),
            action_noise=noise,
            **common,
        )
    if algo == "ddpg":
        return DDPG(
            learning_rate=lr, gamma=gamma,
            batch_size=int(hp.get("batch_size", 256)),
            tau=float(hp.get("tau", 0.005)),
            buffer_size=int(hp.get("buffer_size", 300_000)),
            learning_starts=int(hp.get("learning_starts", 1000)),
            action_noise=noise,
            **common,
        )
    raise ValueError(f"Unknown algorithm: {algo}")


def algo_family(algo: str) -> str:
    return "on_policy" if algo.lower() in ON_POLICY else "off_policy"
