---
name: ml-dl-rl-modeling
description: Train, evaluate, and benchmark Machine Learning (Scikit-Learn), Deep Learning (PyTorch), and Reinforcement Learning (Stable-Baselines3) models for quantitative financial economics, return prediction, regime nowcasting, and dynamic portfolio optimization.
version: 1.0.0
author: AI Quantitative Research Network
tags: [machine-learning, deep-learning, reinforcement-learning, pytorch, stable-baselines3, scikit-learn, statsmodels]
---

# ML / DL / RL Modeling Skill

## Overview
This skill establishes standardized architectures for empirical financial modeling using Machine Learning (`scikit-learn`), Deep Learning (`pytorch`), and Reinforcement Learning (`stable-baselines3`), operating in tandem with econometric and statistical foundations (`statsmodels`, `scipy`, `polars`, `numpy`).

## Standard Library Stack
* **Machine Learning**: `scikit-learn` (Ridge, ElasticNet, RandomForest, GradientBoosting, TimeSeriesSplit).
* **Deep Learning**: `torch` (MLP, LSTM, TCN, attention-based feature extractors).
* **Reinforcement Learning**: `stable_baselines3` (PPO, A2C, SAC) paired with custom Gym environments.
* **Data & Econometrics**: `polars` (zero-copy extraction, PIT backward joins), `statsmodels` (HAC Newey--West benchmarks), `numpy`, `scipy`.

## When to Use
- **ML (Supervised Learning)**: When predicting cross-sectional equity returns, earnings surprises (SUE), or default risk from multidimensional accounting ratios and order flow features (inspired by Paper 010).
- **DL (Deep Representation Learning)**: When modeling high-dimensional non-linear macroeconomic regime transitions or mixed-frequency volatility surfaces (inspired by Paper 003).
- **RL (Policy Optimization)**: When designing dynamic multi-asset rebalancing policies or central bank Taylor rule optimization with transaction costs and drawdown penalties (inspired by Papers 003 and 012).

---

## 1. Machine Learning Recipe (Scikit-Learn Purged TimeSeriesSplit)

```python
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

def train_ml_return_predictor(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Evaluates cross-sectional return predictability using expanding window TimeSeriesSplit
    to prevent look-ahead leakage.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oos_preds = []
    oos_actuals = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        oos_preds.extend(preds)
        oos_actuals.extend(y_test)

    oos_preds = np.array(oos_preds)
    oos_actuals = np.array(oos_actuals)

    ic = np.corrcoef(oos_preds, oos_actuals)[0, 1]
    r2 = r2_score(oos_actuals, oos_preds)
    return {"information_coefficient": float(ic), "r2_oos": float(r2), "n_oos": len(oos_actuals)}
```

---

## 2. Deep Learning Recipe (PyTorch Temporal Volatility Network)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class VolatilityLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Take final temporal step representation
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)

def train_dl_model(X_train_seq: np.ndarray, y_train: np.ndarray, epochs: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VolatilityLSTM(input_dim=X_train_seq.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    return model
```

---

## 3. Deep Reinforcement Learning Recipe (Stable-Baselines3 Portfolio Agent)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class PortfolioAllocationEnv(gym.Env):
    """
    Dynamic portfolio weighting environment with realistic transaction fees.
    """
    def __init__(self, asset_returns: np.ndarray, cost_bps: float = 15.0):
        super().__init__()
        self.returns = asset_returns  # Shape: (T, N_assets)
        self.T, self.N = asset_returns.shape
        self.cost_rate = cost_bps / 10000.0

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.N * 5,), dtype=np.float32)
        self.current_step = 5
        self.weights = np.ones(self.N) / self.N

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 5
        self.weights = np.ones(self.N) / self.N
        obs = self.returns[self.current_step - 5:self.current_step].flatten()
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray):
        raw_weights = np.clip(action, 0, 1)
        new_weights = raw_weights / (np.sum(raw_weights) + 1e-8)

        # Turnover and friction
        turnover = np.sum(np.abs(new_weights - self.weights))
        fee = turnover * self.cost_rate

        # Gross portfolio return
        period_ret = np.dot(new_weights, self.returns[self.current_step])
        net_ret = period_ret - fee

        # Reward: Differential Sharpe / utility
        reward = float(net_ret - 0.5 * (net_ret ** 2))

        self.weights = new_weights
        self.current_step += 1
        done = self.current_step >= self.T - 1

        next_obs = self.returns[self.current_step - 5:self.current_step].flatten()
        return next_obs.astype(np.float32), reward, done, False, {"net_ret": net_ret}

def train_rl_portfolio_agent(asset_returns: np.ndarray, total_timesteps: int = 50000):
    env = PortfolioAllocationEnv(asset_returns)
    agent = PPO("MlpPolicy", env, learning_rate=3e-4, gamma=0.99, verbose=0, seed=42)
    agent.learn(total_timesteps=total_timesteps)
    return agent
```

---

## 4. Empirical Evaluation Protocol for ML/DL/RL
1. **Linear Econometric Benchmark**: Every ML/DL/RL model must be compared directly against an explicit linear baseline (OLS / Newey--West HAC or static 1/N allocation).
2. **Strict Falsifiability**: Formulate the formal Null Hypothesis ($H_0$: ML/DL/RL model out-of-sample Sharpe or Information Ratio does not exceed baseline after accounting for transaction fees).
3. **Out-of-Sample Isolation**: Zero overlapping windows between training, validation, and test datasets.
