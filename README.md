# Progress Knight - AI Enhanced

This project is an AI-Enhanced version and a modified clone of the original [Progress Knight](https://github.com/ihtasham42/progress-knight) by ihtasham42. Credit for the original game goes to ihtasham42.

### Introduction
Progress Knight is a text-based incremental game, developed by Ihtasham42, which can be played on your browser.

### What is Progress Knight about?
Progress Knight is a life-sim incremental based in a fantasy/medieval setting, where you must progress through the career ladder and acquire new skills to become the ultimate being.

You first start off as a beggar, barely being able to feed yourself as the days go by. However, over the years you learn new skills and gain plenty of work experience to enter new high paying jobs while managing your living expenses...

Will you decide to take the easy route of doing simple commoner work? Or will you go through harsh training to climb the ranks within the military? Or maybe will you decide to study hard and enrol in a magic academy, learning life-impacting spells? Your career path is open-ended, the decision is up to you.

Eventually, your age will catch up to you. You will be given an option to prestige and gain xp multipliers (based on the performance of your current life) for your next life at the cost of losing all your levels and assets. Fear not though, as you will re-gain your levels much, much more quickly than in your previous life...

### AI Enhancements

This version integrates an AI agent capable of learning to play Progress Knight autonomously using Reinforcement Learning (RL). Key features include:

*   **Deep Q-Network (DQN) Agent**: The core AI is a DQN agent implemented using TensorFlow.js. It learns optimal strategies through trial-and-error by interacting with the game environment.
*   **Experience Replay**: The agent stores its experiences (state, action, reward, next state) and replays them to efficiently learn the value of different actions in various game states.
*   **Learning from Demonstrations (Behavioral Cloning)**: Players can record their own gameplay sessions. These "demonstrations" can be used to pre-train the agent, giving it a head start by mimicking human strategies before it begins exploring on its own.
*   **Dedicated AI Control Panel**: A new "AI Agent" tab provides comprehensive tools for interacting with the agent:
    *   **Monitoring**: View real-time agent status, key metrics like exploration rate (epsilon), memory usage, recent rewards, and auto-rebirth counts.
    *   **Control**: Pause, resume, save, and load the agent's learned model directly within the browser's local storage.
    *   **Demonstration Management**: Record, save, load, and clear human gameplay demonstrations. Initiate pre-training using loaded demonstrations.
    *   **Hyperparameter Tuning**: Adjust various reward function weights, learning parameters, and interaction settings via the UI to customize the agent's learning process and objectives. Parameters include:
        *   **Reward Weights:**
            *   `Net Income Wgt`: Reward weight for having positive net income.
            *   `Bankruptcy Pen`: Penalty for going bankrupt (large negative).
            *   `Job Progression Bonus`: Bonus for getting a higher-tier job in the same category.
            *   `Income Gain Wgt`: Reward weight for raw income/day increase.
            *   `Surplus Lvl Wgt`: Reward weight for sum of (Level - MaxLevel) increase.
            *   `Item Count Wgt`: Base weight for compounding reward for active misc items (count^2 * weight).
            *   `Item Loss Pen`: Penalty for misc item count decreasing (except on bankruptcy).
            *   `Coin Gain Wgt`: Reward weight for logarithmic increase in coins.
            *   `Evil Gain Wgt`: Reward weight for increasing the Evil resource.
            *   `Unlock Bonus`: Bonus reward per newly unlocked job, skill, or item.
            *   `Death Penalty`: Penalty for hitting the lifespan limit without rebirthing.
        *   **Learning Parameters:**
            *   `Gamma (Discount)`: Discount factor for future rewards (0=myopic, ~1=far-sighted).
            *   `Min Epsilon`: Minimum exploration chance (agent never stops exploring entirely).
            *   `Epsilon Decay`: Multiplier for exploration rate decay per step (e.g., 0.999 = slow decay).
            *   `Learning Rate`: Step size for adjusting neural network weights during training.
            *   `Max Memory Size`: Maximum number of past experiences stored for experience replay.
            *   `Batch Size`: Number of experiences sampled from memory for each training step.
        *   **Interaction:**
            *   `Action Frequency`: Agent makes a decision every N game ticks.
*   **Game Speed Control**: An adjustable game speed multiplier allows for significantly accelerated agent training and observation of long-term strategies.
*   **Customizable Goals**: By tuning the reward parameters, the agent can be guided to prioritize different aspects of the game, such as maximizing income, achieving specific milestones, or balancing exploration and exploitation.

### Where can I play Progress Knight?
Progress Knight can be played on the following sites:  
- [Github Pages](https://ihtasham42.github.io/progress-knight/)  
- [Armor Games](https://armorgames.com/progress-knight-game/19095)
- [Crazy Games](https://www.crazygames.com/game/progress-knight)
