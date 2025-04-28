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
    *   **Hyperparameter Tuning**: Adjust various reward function weights and learning parameters (e.g., discount factor, learning rate, exploration decay, replay memory size) to customize the agent's learning process and objectives.
*   **Game Speed Control**: An adjustable game speed multiplier allows for significantly accelerated agent training and observation of long-term strategies.
*   **Customizable Goals**: By tuning the reward parameters, the agent can be guided to prioritize different aspects of the game, such as maximizing income, achieving specific milestones, or balancing exploration and exploitation.

### Where can I play Progress Knight?
Progress Knight can be played on the following sites:  
- [Github Pages](https://ihtasham42.github.io/progress-knight/)  
- [Armor Games](https://armorgames.com/progress-knight-game/19095)
- [Crazy Games](https://www.crazygames.com/game/progress-knight)
