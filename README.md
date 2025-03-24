# DRL Taxi Assignment

This project implements a reinforcement learning (RL) agent to navigate a dynamic taxi environment inspired by OpenAI Gym’s Taxi-v3. The goal is to train an agent that can efficiently pick up and drop off passengers in a grid-world with randomized configurations, dynamic obstacles, and varying grid sizes.

---

## Project Overview

- **Objective:**  
  Develop an RL agent that adapts to different grid sizes and randomly generated environments while efficiently managing fuel, avoiding obstacles, and handling passenger pickup and drop-off actions.

- **Key Challenge:**  
  The agent must generalize beyond a fixed map. The training and test environments are not identical, and the agent needs to learn strategies that work across varied scenarios.

---

## Environment Details

The custom taxi environment is defined in `simple_custom_taxi_env.py` and includes:

- **Dynamic Grid-World:**  
  - Grid size varies (n × n where n is between 5 and 10).  
  - Four station locations (e.g., R, G, B, Y) represent possible passenger pickup/drop-off points.
  - Random obstacles are placed on the grid. Hitting an obstacle incurs a penalty.

- **Game Mechanics:**  
  - The taxi starts at a random available position.
  - A passenger’s starting location and destination are chosen randomly from the available stations.
  - Actions include moving in four directions, picking up the passenger, and dropping off the passenger.
  - Rewards and penalties are given based on movement efficiency, correct/incorrect pickup/drop-off actions, and obstacle collisions.

- **Fuel Management:**  
  - The taxi has a fuel limit, and each move reduces the available fuel. Running out of fuel ends the game with a penalty.

---

## Agent Implementation

The RL agent is implemented in `student_agent.py` and features:

- **Q-Table Based Approach:**  
  - The agent loads a pre-trained Q-table from `q_table.pkl` (ensure this file is included in your submission if you rely on it).
  - It maps a custom state representation—derived from the taxi’s position, obstacle proximity, and relative positions of passenger and destination—to Q-values for each available action.

- **State Processing & Reward Shaping:**  
  - The function `get_state_obs` creates a state representation that includes directional information, obstacle presence, and the current status of passenger pickup.
  - Reward shaping is applied based on the action taken and environmental conditions (e.g., penalties for hitting obstacles or executing incorrect pickup/drop-off actions).

- **Action Selection:**  
  - The `get_action(obs)` function selects the optimal action using the Q-table.  
  - If the current state is not found in the Q-table, a random action is selected as a fallback.

- **Q-Table Update:**  
  - The agent updates its Q-table values in real time using the reward feedback, promoting improved decision-making over time.

---

## Requirements

Install these dependencies with:
```bash
pip install -r requirements.txt
```
