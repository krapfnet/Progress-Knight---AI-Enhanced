// TensorFlow.js will be used for the neural network and learning algorithms.
// We'll need to import it in the HTML file later.

const PARAMS_STORAGE_KEY = 'progressKnightAgentParams'; // Moved to top
const DEMOS_STORAGE_KEY = 'progressKnightAgentDemos'; // Moved to top

// --- Agent Parameters (Defaults & Current Values) ---
let agentParams = {
    // Rewards
    positiveNetIncomeWeight: 0.2,  // Increased default
    bankruptcyPenalty: -150,      
    jobProgressionBonus: 20.0, 
    incomeGainWeight: 0.01,      // Keeping low, emphasize Net
    surplusLevelGainWeight: 1.0, 
    itemCountWeight: 0.2,        // Increased default
    itemLossPenalty: 10.0,       
    coinGainLogWeight: 0.1,      // Increased default
    evilGainWeight: 1.0,         
    unlockBonus: 15.0,           
    deathPenalty: 0,             // Changed default to 0
    // Learning
    gamma: 0.99,           
    epsilonMin: 0.05,       
    epsilonDecay: 0.999,     
    learningRate: 0.0005,    
    maxMemorySize: 10000,     
    agentBatchSize: 32,      
    // Interaction
    agentActionFrequency: 25 // Increased default
};

// --- Global Demonstration Storage ---
let demonstrations = [];
let isRecording = false;
// const DEMOS_STORAGE_KEY = 'progressKnightAgentDemos'; // Removed from here

class RLAgent {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;

        // --- RL Parameters (using global agentParams) ---
        this.memory = []; // Replay buffer
        this.maxMemorySize = agentParams.maxMemorySize; 
        this.gamma = agentParams.gamma;    
        this.epsilon = 1.0;  // Epsilon starts at 1.0, decays towards epsilonMin
        this.epsilonMin = agentParams.epsilonMin; 
        this.epsilonDecay = agentParams.epsilonDecay; 
        this.learningRate = agentParams.learningRate; 
        this.isTraining = false; // Added training lock flag

        // --- Neural Network Model (using TensorFlow.js) ---
        this.model = this._buildModel(); // learningRate is used here
    }

    /**
     * Builds the neural network model using TensorFlow.js.
     * @private
     */
    _buildModel() {
        // Define the model architecture (Requires tf object)
        if (typeof tf === 'undefined') {
            console.error("TensorFlow.js (tf) not loaded.");
            return null;
        }
        const model = tf.sequential();
        // Input layer: state features
        model.add(tf.layers.dense({units: 128, activation: 'relu', inputShape: [this.stateSize]})); // Increased complexity
        model.add(tf.layers.dense({units: 128, activation: 'relu'}));
        // Output layer: Q-values for each action
        model.add(tf.layers.dense({units: this.actionSize, activation: 'linear'})); // Linear activation for Q-values

        model.compile({
            loss: 'meanSquaredError', // Common loss function for Q-learning
            optimizer: tf.train.adam(this.learningRate), // Use agent's current learning rate
            metrics: ['accuracy'] // Optional: track accuracy during training
        });
        console.log("RL Agent Model Built:", model.summary());
        return model;
    }

    /**
     * Stores a transition in the replay memory, managing its size.
     */
    remember(state, action, reward, nextState, done) {
        // Use agentParams for max size check
        if (this.memory.length > agentParams.maxMemorySize) {
            this.memory.shift(); 
        }
        this.memory.push({ state, action, reward, nextState, done });
    }

    /**
     * Chooses an action based on the current state using an epsilon-greedy strategy.
     * Also returns the Q-values for debugging/analysis if needed.
     */
    act(state, availableActionsMask) {
        if (!this.model || typeof tf === 'undefined') {
            console.error("Model not initialized or TensorFlow.js not available!");
            return this.getRandomAvailableAction(availableActionsMask); // Fallback
        }

        // Epsilon-greedy exploration/exploitation
        if (Math.random() <= this.epsilon) {
            // Explore: choose a random *available* action
            return this.getRandomAvailableAction(availableActionsMask);
        } else {
            // Exploit: choose the best available action predicted by the model
            return tf.tidy(() => {
                 const stateTensor = tf.tensor2d(state, [1, this.stateSize]);
                 const allQValues = this.model.predict(stateTensor);

                 // Apply the mask: Set Q-values of unavailable actions to a very low number
                 const maskedQValues = tf.where(
                    tf.tensor1d(availableActionsMask, 'bool'),
                    allQValues,
                    tf.fill(allQValues.shape, -Infinity) // Set unavailable actions to -Infinity
                 );

                 const actionIndex = tf.argMax(maskedQValues, 1).dataSync()[0];
                 // stateTensor.dispose(); // tf.tidy handles disposal
                 // allQValues.dispose();
                 // maskedQValues.dispose();
                 return actionIndex;
            });
        }
    }

    /**
     * Helper to get a random action from the available ones.
     */
    getRandomAvailableAction(availableActionsMask) {
        const availableIndices = [];
        for(let i = 0; i < availableActionsMask.length; i++) {
            if (availableActionsMask[i]) {
                availableIndices.push(i);
            }
        }
        if (availableIndices.length === 0) {
            console.warn("No available actions to choose from!");
            return 0; // Fallback: return first action index
        }
        const randomIndex = Math.floor(Math.random() * availableIndices.length);
        return availableIndices[randomIndex];
    }

    /**
     * Trains the neural network using a batch of experiences from the replay memory (DQN).
     */
    async replay(batchSize) {
        // Check training lock
        if (this.isTraining) {
            // console.log("Replay skipped: Training already in progress.");
            return; 
        }
        
        if (!this.model || typeof tf === 'undefined' || this.memory.length < batchSize) {
            return; 
        }

        // Sample a minibatch from memory
        const minibatch = [];
        const memoryIndices = new Set();
        while(minibatch.length < Math.min(batchSize, this.memory.length)) {
            let index = Math.floor(Math.random() * this.memory.length);
            if (!memoryIndices.has(index)) {
                 minibatch.push(this.memory[index]);
                 memoryIndices.add(index);
            }
        }

        // Extract data from minibatch
        const states = minibatch.map(mem => mem.state);
        const actions = minibatch.map(mem => mem.action);
        const rewards = minibatch.map(mem => mem.reward);
        const nextStates = minibatch.map(mem => mem.nextState);
        const dones = minibatch.map(mem => mem.done);

        // Define tensors outside try block for disposal
        let stateTensor, nextStateTensor, currentQValuesTensor, nextQValuesTensor, targetQValuesTensor;

        try {
            this.isTraining = true; // Set training lock

            // Convert to tensors
            stateTensor = tf.tensor2d(states, [minibatch.length, this.stateSize]);
            nextStateTensor = tf.tensor2d(nextStates, [minibatch.length, this.stateSize]);

            // Predict Q-values for current states and next states
            currentQValuesTensor = this.model.predict(stateTensor);
            nextQValuesTensor = this.model.predict(nextStateTensor);

            // Clone current Q-values to update targets
            const targetQValues = currentQValuesTensor.clone().arraySync(); 
            const nextQValuesData = nextQValuesTensor.arraySync(); // Get data for sync loop

            // Calculate target Q-values using the Bellman equation
            for (let i = 0; i < minibatch.length; i++) {
                let target = rewards[i];
                if (!dones[i]) {
                    // Find the max Q-value for the next state from sync data
                    const maxNextQ = Math.max(...nextQValuesData[i]); 
                    target += this.gamma * maxNextQ;
                }
                // Update the Q-value for the action actually taken
                targetQValues[i][actions[i]] = target;
            }

            // Convert target Q-values back to tensor
            targetQValuesTensor = tf.tensor2d(targetQValues, [minibatch.length, this.actionSize]);

            // Train the model
            const history = await this.model.fit(stateTensor, targetQValuesTensor, {
                epochs: 1, 
                verbose: 0 
            });
             // console.log("Replay Training Loss:", history.history.loss[0]);
        } catch(err) {
             console.error("Error during replay:", err);
        } finally {
             // Manual Tensor Disposal
             if (stateTensor) stateTensor.dispose();
             if (nextStateTensor) nextStateTensor.dispose();
             if (currentQValuesTensor) currentQValuesTensor.dispose();
             if (nextQValuesTensor) nextQValuesTensor.dispose();
             if (targetQValuesTensor) targetQValuesTensor.dispose();
             // console.log("Disposed replay tensors.");
             
             this.isTraining = false; // Release training lock
        }
        
        // Decay epsilon after training
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    /**
     * Load pre-trained model weights from local storage.
     */
    async load(modelPath = 'localstorage://progress-knight-agent') {
        if (!this.model || typeof tf === 'undefined') {
             console.error("Model not initialized or TF.js not available for loading.");
             return;
        }
        try {
            console.log(`Loading model from ${modelPath}...`);
            this.model = await tf.loadLayersModel(modelPath);
            // Re-compile the model after loading (important!)
             this.model.compile({
                loss: 'meanSquaredError',
                optimizer: tf.train.adam(this.learningRate), // Use agent's current LR
                metrics: ['accuracy']
            });
            console.log("Model loaded successfully.");
            // Optionally, reset epsilon if loading a trained model
            // this.epsilon = this.epsilonMin; // Start exploiting more if loaded
        } catch (err) {
            console.log(`Failed to load model from ${modelPath}. Starting fresh or continuing training.`, err);
        }
    }

    /**
     * Save current model weights to local storage.
     */
    async save(modelPath = 'localstorage://progress-knight-agent') {
        if (!this.model || typeof tf === 'undefined') {
            console.error("Model not initialized or TF.js not available for saving.");
            return;
        }
        try {
            console.log(`Saving model to ${modelPath}...`);
            const saveResult = await this.model.save(modelPath);
            console.log("Model saved successfully:", saveResult);
        } catch (err) {
            console.error(`Failed to save model to ${modelPath}.`, err);
        }
    }

    /**
     * Pre-trains the model on human demonstrations using supervised learning.
     * @param {Array} demonstrationsData - Array of {state, actionIndex} objects.
     * @param {number} epochs - Number of training epochs.
     */
    async pretrain(demonstrationsData, epochs = 5) {
        if (!this.model || typeof tf === 'undefined') {
            console.error("Model not ready for pretraining.");
            alert("Model not ready for pretraining.");
            return;
        }
        if (!demonstrationsData || demonstrationsData.length === 0) {
            console.warn("No demonstration data provided for pretraining.");
            alert("No demonstration data provided for pretraining.");
            return;
        }

        console.log(`Starting pretraining on ${demonstrationsData.length} demonstrations for ${epochs} epochs...`);
        alert(`Starting pretraining on ${demonstrationsData.length} demonstrations for ${epochs} epochs. This might take a moment and may freeze the browser tab.`);

        const originalOptimizer = tf.train.adam(this.learningRate);
        const originalLoss = 'meanSquaredError';
        const originalActivation = 'linear'; // Assuming last layer is linear for DQN

        let statesTensor, actionsTensor, actionsOneHotTensor;

        try {
            const states = demonstrationsData.map(d => d.state);
            const actions = demonstrationsData.map(d => d.actionIndex);

            statesTensor = tf.tensor2d(states, [states.length, this.stateSize]);
            actionsTensor = tf.tensor1d(actions, 'int32'); 
            
            // Convert actions to one-hot encoding
            actionsOneHotTensor = tf.oneHot(actionsTensor, this.actionSize).toFloat(); // Depth is actionSize

            console.log("Pretrain States Tensor Shape:", statesTensor.shape);
            console.log("Pretrain Actions OneHot Shape:", actionsOneHotTensor.shape);
            
            // Ensure states tensor is float32
            const statesTensorFloat = statesTensor.toFloat(); 

            // 3. Compile model for classification with categoricalCrossentropy
            // Requires softmax activation on the output layer for probabilities.
            // Temporarily rebuild or modify the last layer? Less ideal.
            // Alternatively, specify fromLogits=true for categoricalCrossentropy if supported?
            // Let's try compiling with categoricalCrossentropy and see if TF handles it.
            // NOTE: The underlying model's last layer is still 'linear'. 
            // This loss expects probabilities (softmax output). TF might handle logits implicitly,
            // but this mismatch could be problematic.
             this.model.compile({
                loss: 'categoricalCrossentropy', // Changed loss
                optimizer: tf.train.adam(this.learningRate), 
                metrics: ['accuracy']
            });
            console.log("Model recompiled for pretraining with categoricalCrossentropy.");

            // 4. Train using model.fit() with one-hot actions
            const history = await this.model.fit(statesTensorFloat, actionsOneHotTensor, { 
                epochs: epochs,
                batchSize: agentParams.agentBatchSize || 32, 
                shuffle: true,
                validationSplit: 0.1, 
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                         console.log(`Pretrain Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}, Acc: ${logs.acc.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}, Val Acc: ${logs.val_acc.toFixed(4)}`);
                    }
                }
            });
            console.log("Pretraining fit complete.", history.history);

        } catch (err) {
            console.error("Error during pretraining:", err); 
            alert(`Error during pretraining: ${err.message || err}. Check console.`);
        } finally {
             // Manual Tensor Disposal
             if (statesTensor) statesTensor.dispose();
             if (statesTensorFloat) statesTensorFloat.dispose(); 
             if (actionsTensor) actionsTensor.dispose();
             if (actionsOneHotTensor) actionsOneHotTensor.dispose(); // Dispose one-hot tensor
             console.log("Disposed pretraining tensors.");
            
             // Restore original model compilation for RL
             this.model.compile({
                loss: originalLoss, 
                optimizer: originalOptimizer,
                metrics: ['accuracy'] 
            });
            console.log("Model recompiled for RL fine-tuning.");
             // Modify alert to reflect potential success or failure
             // Alert is now inside the catch block on failure
             if (!err) { // Check if error occurred (less ideal, better flag needed ideally)
                 alert("Pretraining finished.");
             } 
        }
    }
}

// --- Interface Functions ---

/**
 * Extracts the current game state and converts it into a normalized numerical array.
 * @returns {Array} The numerical representation of the game state.
 */
function getAgentState() {
    // Ensure game data is available
    if (typeof gameData === 'undefined' || typeof getLifespan === 'undefined' || !gameData.taskData || !gameData.itemData) {
        console.error("Cannot get agent state: game data not ready.");
        return new Array(stateSize).fill(0); // Return dummy state
    }

    const state = [];

    // 1. Core Resources (Normalized)
    // Use log scale for potentially large values like coins
    state.push(normalizeLog(gameData.coins));
    const currentLifespan = getLifespan();
    state.push(normalize(gameData.days, 0, currentLifespan * 1.1)); // Normalize days based on lifespan
    state.push(normalizeLog(gameData.evil)); // Log scale for evil?
    state.push(normalize(gameData.rebirthOneCount, 0, 10)); // Assume max 10 rebirths for normalization
    state.push(normalize(gameData.rebirthTwoCount, 0, 5)); // Assume max 5 evil rebirths

    // 2. Task Progress (Level and Normalized XP)
    // Ensure consistent order (e.g., alphabetical by name)
    const taskNames = Object.keys(gameData.taskData).sort();
    for (const taskName of taskNames) {
        const task = gameData.taskData[taskName];
        state.push(normalizeLog(task.level)); // Log scale for potentially high levels
        const maxXp = task.getMaxXp ? task.getMaxXp() : 1; // Handle potential undefined method briefly
        state.push(normalize(task.xp, 0, maxXp));
    }

    // 3. Property Ownership (One-hot encoding)
    const properties = itemCategories["Properties"].sort();
    for (const propName of properties) {
        state.push(gameData.currentProperty && gameData.currentProperty.name === propName ? 1 : 0);
    }

    // 4. Misc Ownership (Multi-hot encoding)
    const miscItems = itemCategories["Misc"].sort();
    const ownedMiscNames = new Set(gameData.currentMisc.map(item => item.name));
    for (const miscName of miscItems) {
        state.push(ownedMiscNames.has(miscName) ? 1 : 0);
    }

    // Check if state size matches expected
    if (state.length !== stateSize) {
         console.warn(`Generated state size (${state.length}) does not match expected state size (${stateSize}). Check calculations.`);
         // Pad or truncate if necessary (though ideally calculations should be correct)
         while (state.length < stateSize) state.push(0);
         if (state.length > stateSize) state.length = stateSize;
    }

    return state;
}

/**
 * Creates a boolean mask indicating which actions are currently valid.
 * @returns {Array<boolean>} Mask array where true means the action is valid.
 */
function getAvailableActionsMask() {
    const mask = new Array(actionSize).fill(false); // Start with all actions unavailable

    if (typeof gameData === 'undefined' || typeof requirementsMet === 'undefined') {
        console.error("Cannot get available actions: game data not ready.");
        return mask;
    }

    for (let i = 0; i < actionMap.length; i++) {
        const action = actionMap[i];
        if (!action) continue; // Skip if map isn't fully built

        try {
            switch (action.type) {
                case 'setTask':
                    // Check if the job/skill exists and its requirements are met
                    if (gameData.taskData[action.name] && requirementsMet(action.name)) {
                        mask[i] = true;
                    }
                    break;
                case 'setProperty':
                    // Check if property exists, requirements met, and affordable
                    const property = gameData.itemData[action.name];
                    if (property && requirementsMet(action.name) && gameData.coins >= property.getExpense()) {
                        mask[i] = true;
                    }
                    break;
                case 'setMisc':
                    // Check if misc item exists, requirements met
                    const misc = gameData.itemData[action.name];
                    if (misc && requirementsMet(action.name)) {
                        // Allow buying if affordable, allow selling if owned
                        const isOwned = gameData.currentMisc.some(m => m.name === action.name);
                        if (isOwned || gameData.coins >= misc.getExpense()) {
                             mask[i] = true;
                        }
                    }
                    break;
                case 'rebirthOne':
                    // Check if rebirth 1 requirements met (e.g., age)
                    // Need a function like `canRebirthOne()` or check requirements directly
                    if (requirementsMet("Rebirth note 2")) { // Approximation based on notes
                        mask[i] = true;
                    }
                    break;
                case 'rebirthTwo':
                    // Check if rebirth 2 requirements met (e.g., age, maybe evil)
                    if (requirementsMet("Rebirth note 3")) { // Approximation
                         mask[i] = true;
                    }
                    break;
            }
        } catch (e) {
             console.error(`Error checking availability for action ${i} (${action.name}):`, e);
        }
    }
    return mask;
}

// Helper to check requirements (needs access to gameData.requirements)
function requirementsMet(entityName) {
    if (!gameData.requirements || !gameData.requirements[entityName]) {
        // If entity has no requirements object, assume it's available (like 'Beggar')
        // Or if the requirement itself is missing somehow
        return !!(gameData.taskData[entityName] || gameData.itemData[entityName]);
    }
    return gameData.requirements[entityName].isCompleted();
}


/**
 * Maps an action index from the agent to a specific game function call, checking validity.
 * @param {number} actionIndex - The index of the action chosen by the agent.
 */
function performAgentAction(actionIndex) {
    if (actionIndex < 0 || actionIndex >= actionMap.length || !actionMap[actionIndex]) {
        console.error(`Invalid action index received: ${actionIndex}`);
        return;
    }

    const action = actionMap[actionIndex];

    // Re-check validity just before executing (state might change slightly)
    const mask = getAvailableActionsMask();
    if (!mask[actionIndex]) {
        // console.warn(`Agent chose unavailable action ${actionIndex} (${action.name}). Doing nothing.`);
        return; // Don't perform an invalid action
    }

    // Perform the action
    try {
        // console.log(`Agent performing action: ${action.type} - ${action.name || ''}`);
        switch (action.type) {
            case 'setTask':
                // Check if it's already the current task to avoid redundant calls
                const currentTask = action.category === 'job' ? gameData.currentJob : gameData.currentSkill;
                if (!currentTask || currentTask.name !== action.name) {
                    setTask(action.name); // Call global game function
                }
                break;
            case 'setProperty':
                 if (gameData.currentProperty.name !== action.name) {
                    setProperty(action.name); // Call global game function
                 }
                break;
            case 'setMisc':
                setMisc(action.name); // Call global game function (toggles)
                break;
            case 'rebirthOne':
                rebirthOne(); // Call global game function
                break;
            case 'rebirthTwo':
                rebirthTwo(); // Call global game function
                break;
            default:
                console.warn(`Unknown action type in action map: ${action.type}`);
        }
    } catch (e) {
        console.error(`Error executing action ${actionIndex} (${action.name}):`, e);
    }
}

// --- Reward Calculation ---
let lastReward = 0; // Store the last calculated reward for UI display

// Helper function to calculate sum of levels exceeding max levels
function calculateTotalSurplusLevel() {
    let totalSurplus = 0;
    if (gameData && gameData.taskData) {
        for (const taskName in gameData.taskData) {
            const task = gameData.taskData[taskName];
            if (task.level > task.maxLevel) { // maxLevel should exist on tasks
                totalSurplus += (task.level - task.maxLevel);
            }
        }
    }
    return totalSurplus;
}

function captureStateSnapshot() {
    // Capture key metrics needed for reward calculation
    let totalSkill = 0;
    const unlockedEntities = new Set(); // Set to store names of unlocked entities

    if (gameData && gameData.taskData) {
         for (const taskName in gameData.taskData) {
            if (typeof skillBaseData !== 'undefined' && skillBaseData[taskName]) {
                 totalSkill += gameData.taskData[taskName].level;
            }
            // Check if task is unlocked
            if (typeof requirementsMet === 'function' && requirementsMet(taskName)) {
                unlockedEntities.add(taskName);
            }
        }
    }
    if (gameData && gameData.itemData) {
        for (const itemName in gameData.itemData) {
            // Check if item is unlocked
             if (typeof requirementsMet === 'function' && requirementsMet(itemName)) {
                unlockedEntities.add(itemName);
            }
        }
    }

    const currentIncome = (typeof getIncome === 'function') ? getIncome() : 0;
    const currentExpense = (typeof getExpense === 'function') ? getExpense() : 0; // Need getExpense
    const actualNetIncome = currentIncome - currentExpense; // Calculate actual net
    const currentPropertyName = (gameData && gameData.currentProperty) ? gameData.currentProperty.name : "";
    const currentJobName = (gameData && gameData.currentJob) ? gameData.currentJob.name : ""; // Added

    return {
        totalSkillLevel: totalSkill, 
        totalSurplusLevel: calculateTotalSurplusLevel(), 
        income: currentIncome, 
        actualNetIncome: actualNetIncome,
        property: currentPropertyName,
        jobName: currentJobName, // Added
        evil: gameData ? gameData.evil : 0,
        days: gameData ? gameData.days : 0,
        coins: gameData ? gameData.coins : 0, 
        miscItemCount: gameData ? gameData.currentMisc.length : 0, 
        unlockedEntities: unlockedEntities, // Added set of unlocked entity names
        isAlive: typeof isAlive !== 'undefined' ? isAlive() : true
    };
}

/**
 * Calculates the reward signal based on the change between state snapshots.
 * @param {object} previousStateSnapshot - Snapshot before the action/step.
 * @param {object} currentStateSnapshot - Snapshot after the action/step.
 * @returns {number} The calculated reward.
 */
function calculateReward(previousStateSnapshot, currentStateSnapshot) {
    let reward = 0;
    
    // --- Parameters for tuning ---
    const positiveNetIncomeWeight = agentParams.positiveNetIncomeWeight; 
    const bankruptcyPenalty = agentParams.bankruptcyPenalty;       
    const surplusLevelGainWeight = agentParams.surplusLevelGainWeight; 
    const itemCountWeight = agentParams.itemCountWeight;        
    const itemLossPenalty = agentParams.itemLossPenalty;       
    const coinGainLogWeight = agentParams.coinGainLogWeight;      
    const evilGainWeight = agentParams.evilGainWeight;         
    const unlockBonus = agentParams.unlockBonus;           
    const jobProgressionBonus = agentParams.jobProgressionBonus; // Added
    const deathPenalty = agentParams.deathPenalty;          

    // --- Calculate Rewards --- 

    // 1. Positive Net Income
    if (currentStateSnapshot.actualNetIncome > 0) {
        // Give a reward scaled by the net income (normalize?)
        // Simple positive reward for now, scaled slightly by magnitude
        reward += Math.log10(currentStateSnapshot.actualNetIncome + 1) * positiveNetIncomeWeight;
    }

    // 2. Bankruptcy Penalty
    if (previousStateSnapshot.property !== 'Homeless' && currentStateSnapshot.property === 'Homeless') {
        reward += bankruptcyPenalty; 
        console.log("Agent went bankrupt! Applying penalty.");
    }

    // 3. Job Progression Bonus (NEW)
    if (currentStateSnapshot.jobName && previousStateSnapshot.jobName && 
        currentStateSnapshot.jobName !== previousStateSnapshot.jobName) {
        let prevCategory, prevIndex = -1, currentCategory, currentIndex = -1;
        // Find categories and indices (assuming jobCategories is global)
        for (const category in jobCategories) {
            const prevJobIndex = jobCategories[category].indexOf(previousStateSnapshot.jobName);
            const currentJobIndex = jobCategories[category].indexOf(currentStateSnapshot.jobName);
            if (prevJobIndex !== -1) { 
                prevCategory = category; 
                prevIndex = prevJobIndex; 
            }
            if (currentJobIndex !== -1) { 
                currentCategory = category; 
                currentIndex = currentJobIndex; 
            }
        }
        // If jobs are in the same category and index increased, give bonus
        if (prevCategory && currentCategory && prevCategory === currentCategory && currentIndex > prevIndex) {
            reward += jobProgressionBonus;
            // console.log(`Agent progressed job from ${previousStateSnapshot.jobName} to ${currentStateSnapshot.jobName}! Bonus: ${jobProgressionBonus}`);
        }
    }

    // 4. Surplus Level Gain
    const surplusLevelGain = currentStateSnapshot.totalSurplusLevel - previousStateSnapshot.totalSurplusLevel;
    if (surplusLevelGain > 0) {
        reward += surplusLevelGain * surplusLevelGainWeight;
    }

    // 5. Item Count Reward
    const currentItemCount = currentStateSnapshot.miscItemCount;
    if (currentItemCount > 0) {
        reward += (currentItemCount * currentItemCount) * itemCountWeight;
    }

    // 6. Item Loss Penalty
    if (currentStateSnapshot.miscItemCount < previousStateSnapshot.miscItemCount && currentStateSnapshot.property !== 'Homeless') {
        // Only apply penalty if not due to bankruptcy (which has its own penalty)
        reward -= itemLossPenalty;
    }

    // 7. Unlock Bonus
    const newlyUnlocked = new Set([...currentStateSnapshot.unlockedEntities].filter(x => !previousStateSnapshot.unlockedEntities.has(x)));
    if (newlyUnlocked.size > 0) {
        reward += newlyUnlocked.size * unlockBonus;
    }

    // 8. Coin Gain (Log)
    const logCoinGain = Math.log10(currentStateSnapshot.coins + 1) - Math.log10(previousStateSnapshot.coins + 1);
    if (logCoinGain > 0) {
        reward += logCoinGain * coinGainLogWeight;
    }

    // 9. Evil Gain
    const evilGain = currentStateSnapshot.evil - previousStateSnapshot.evil;
    if (evilGain > 0) {
        reward += evilGain * evilGainWeight;
    }
     
    // 10. Penalty for Dying (if deathPenalty !== 0)
    if (!currentStateSnapshot.isAlive && previousStateSnapshot.isAlive) {
        reward += deathPenalty; 
        console.log("Agent died! Applying penalty.");
    }

    lastReward = reward; 
    return reward;
}

// --- Helper functions ---
function normalize(value, min = 0, max = 1) {
     if (max === min) return 0;
     return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function normalizeLog(value, maxLog = 15) { // Log base 10 normalization, maxLog defines scale
     if (value <= 0) return 0;
     const logVal = Math.log10(value);
     return Math.max(0, Math.min(1, logVal / maxLog));
}

// --- Global Agent Instance & Control ---
let agent = null;
let stateSize = 0;
let actionSize = 0;
let actionMap = [];
let isAgentPaused = false; // Flag to control agent activity
let autoRebirthCounter = 0; // Counter for automatic rebirths this session

// --- Parameter Storage ---
// const PARAMS_STORAGE_KEY = 'progressKnightAgentParams'; // Removed duplicate

// Helper to get input element value (parsing as float or int)
function getParamInput(id, isInt = false) {
    const input = document.getElementById(id);
    if (!input) return null;
    const value = isInt ? parseInt(input.value, 10) : parseFloat(input.value);
    return isNaN(value) ? null : value;
}

// Helper to set input element value
function setParamInput(id, value) {
    const input = document.getElementById(id);
    if (input) {
        input.value = value;
    }
}

function saveAgentParams() {
    console.log("Saving agent parameters...");
    try {
        // Read values from inputs and update the agentParams object
        agentParams.positiveNetIncomeWeight = getParamInput('paramNetIncome') ?? agentParams.positiveNetIncomeWeight;
        agentParams.bankruptcyPenalty = getParamInput('paramBankruptcy') ?? agentParams.bankruptcyPenalty;
        agentParams.jobProgressionBonus = getParamInput('paramJobProgressionBonus') ?? agentParams.jobProgressionBonus;
        agentParams.incomeGainWeight = getParamInput('paramIncomeGain') ?? agentParams.incomeGainWeight;
        agentParams.surplusLevelGainWeight = getParamInput('paramSurplusLevel') ?? agentParams.surplusLevelGainWeight;
        agentParams.itemCountWeight = getParamInput('paramItemCount') ?? agentParams.itemCountWeight;
        agentParams.itemLossPenalty = getParamInput('paramItemLoss') ?? agentParams.itemLossPenalty;
        agentParams.coinGainLogWeight = getParamInput('paramCoinGain') ?? agentParams.coinGainLogWeight;
        agentParams.evilGainWeight = getParamInput('paramEvilGain') ?? agentParams.evilGainWeight;
        agentParams.unlockBonus = getParamInput('paramUnlockBonus') ?? agentParams.unlockBonus;
        agentParams.deathPenalty = getParamInput('paramDeathPenalty') ?? agentParams.deathPenalty;
        agentParams.gamma = getParamInput('paramGamma') ?? agentParams.gamma;
        agentParams.epsilonMin = getParamInput('paramEpsilonMin') ?? agentParams.epsilonMin;
        agentParams.epsilonDecay = getParamInput('paramEpsilonDecay') ?? agentParams.epsilonDecay;
        agentParams.learningRate = getParamInput('paramLearningRate') ?? agentParams.learningRate;
        agentParams.maxMemorySize = getParamInput('paramMaxMemory', true) ?? agentParams.maxMemorySize;
        agentParams.agentBatchSize = getParamInput('paramBatchSize', true) ?? agentParams.agentBatchSize;
        agentParams.agentActionFrequency = getParamInput('paramActionFreq', true) ?? agentParams.agentActionFrequency;

        // Save the updated object to localStorage
        localStorage.setItem(PARAMS_STORAGE_KEY, JSON.stringify(agentParams));
        console.log("Agent parameters saved to localStorage.");
        alert("Agent parameters saved!"); 

        // Apply necessary changes immediately (like learning rate, memory size)
        if (agent) {
            agent.learningRate = agentParams.learningRate;
            agent.gamma = agentParams.gamma;
            agent.epsilonMin = agentParams.epsilonMin;
            agent.epsilonDecay = agentParams.epsilonDecay;
            // Recompile model if LR changed?
            // agent._buildModel(); // Careful, this resets weights! Better to just update optimizer if TF allows.
        }

    } catch (e) {
        console.error("Error saving agent parameters:", e);
        alert("Error saving parameters. Check console.");
    }
}

function loadAgentParams() {
    console.log("Loading agent parameters...");
    try {
        const savedParams = localStorage.getItem(PARAMS_STORAGE_KEY);
        if (savedParams) {
            const loaded = JSON.parse(savedParams);
            agentParams = {...agentParams, ...loaded};
            console.log("Agent parameters loaded from localStorage.");
        } else {
            console.log("No saved parameters found, using defaults.");
            localStorage.setItem(PARAMS_STORAGE_KEY, JSON.stringify(agentParams));
        }
        updateParamsUI(); 
    } catch (e) {
        console.error("Error loading agent parameters:", e);
    }
}

// Update the parameter input fields in the UI
function updateParamsUI() {
    setParamInput('paramNetIncome', agentParams.positiveNetIncomeWeight);
    setParamInput('paramBankruptcy', agentParams.bankruptcyPenalty);
    setParamInput('paramJobProgressionBonus', agentParams.jobProgressionBonus);
    setParamInput('paramIncomeGain', agentParams.incomeGainWeight);
    setParamInput('paramSurplusLevel', agentParams.surplusLevelGainWeight);
    setParamInput('paramItemCount', agentParams.itemCountWeight);
    setParamInput('paramItemLoss', agentParams.itemLossPenalty);
    setParamInput('paramCoinGain', agentParams.coinGainLogWeight);
    setParamInput('paramEvilGain', agentParams.evilGainWeight);
    setParamInput('paramUnlockBonus', agentParams.unlockBonus);
    setParamInput('paramDeathPenalty', agentParams.deathPenalty);
    setParamInput('paramGamma', agentParams.gamma);
    setParamInput('paramEpsilonMin', agentParams.epsilonMin);
    setParamInput('paramEpsilonDecay', agentParams.epsilonDecay);
    setParamInput('paramLearningRate', agentParams.learningRate);
    setParamInput('paramMaxMemory', agentParams.maxMemorySize);
    setParamInput('paramBatchSize', agentParams.agentBatchSize);
    setParamInput('paramActionFreq', agentParams.agentActionFrequency);
    // Also update the max memory display span
    const maxMemoryEl = document.getElementById('aiMaxMemorySizeDisplay');
    if (maxMemoryEl) maxMemoryEl.textContent = agentParams.maxMemorySize;
}

function initializeAgent() {
    // Ensure game data is loaded (this function should be called after main.js setup)
    if (typeof jobBaseData === 'undefined' || typeof skillBaseData === 'undefined' || typeof itemCategories === 'undefined' || typeof gameData === 'undefined') {
        console.error("Game data not available for agent initialization. Ensure initializeAgent() is called after main.js loads data.");
        return;
    }

    loadAgentParams(); // Load params before initializing agent
    loadDemonstrations(); // Load demos on init

    console.log("Initializing RL Agent...");
    stateSize = calculateStateSize();
    actionSize = calculateActionSize();
    buildActionMap(); // Build the map after calculating size

    if (stateSize === 0 || actionSize === 0) {
        console.error("Failed to calculate valid state/action sizes.");
        return;
    }

    agent = new RLAgent(stateSize, actionSize);
    console.log(`Agent initialized with State Size: ${stateSize}, Action Size: ${actionSize}`);
    isAgentPaused = false; // Start active
    autoRebirthCounter = 0; // Reset counter on init

    // Reset reward tracking state on init
    // previousTotalSkillLevel = captureStateSnapshot().totalSkillLevel; // Handled by currentStateSnapshot logic now
    // previousEvil = captureStateSnapshot().evil;
    currentStateSnapshot = null; // Reset snapshot
    lastReward = 0;

    // Try loading a saved model upon initialization
    agent.load().catch(err => {
        console.log("No saved model found or error loading:", err);
    });

    // Update button states initially
    updateAgentControlButtonStates();
    updateParamsUI(); // Ensure UI matches loaded params initially
    updateDemoCountUI(); // Initial update
}

function calculateStateSize() {
    let size = 0;
    try {
        const numJobs = Object.keys(jobBaseData).length;
        const numSkills = Object.keys(skillBaseData).length;

        // Core resources
        size += 5; // coins, normalized days, evil, rebirth1, rebirth2

        // Task progress (level, normalized XP) - Ensure consistent order!
        size += (numJobs + numSkills) * 2;

        // Property ownership (one-hot) - Ensure consistent order!
        size += itemCategories["Properties"].length;

        // Misc ownership (multi-hot) - Ensure consistent order!
        size += itemCategories["Misc"].length;

    } catch (e) {
        console.error("Error calculating state size. Is game data loaded?", e);
        return 0;
    }
    return size;
}

function calculateActionSize() {
    let size = 0;
    try {
        const numJobs = Object.keys(jobBaseData).length;
        const numSkills = Object.keys(skillBaseData).length;
        const numProperties = itemCategories["Properties"].length;
        const numMisc = itemCategories["Misc"].length;

        size += numJobs;       // Actions: Set Job
        size += numSkills;     // Actions: Set Skill
        size += numProperties; // Actions: Set Property
        size += numMisc;       // Actions: Toggle Misc Item
        size += 2;             // Actions: Rebirth One, Rebirth Two

    } catch (e) {
        console.error("Error calculating action size. Is game data loaded?", e);
        return 0;
    }
    return size;
}

/**
 * Builds a map correlating the flat action index to the actual action type and name.
 * Ensures consistent ordering based on sorted names/categories.
 */
function buildActionMap() {
    actionMap = new Array(actionSize);
    let index = 0;
    try {
        // Job Actions (Sorted)
        const jobNames = Object.keys(jobBaseData).sort();
        for (const jobName of jobNames) {
            actionMap[index++] = { type: 'setTask', name: jobName, category: 'job' };
        }
        // Skill Actions (Sorted)
        const skillNames = Object.keys(skillBaseData).sort();
        for (const skillName of skillNames) {
            actionMap[index++] = { type: 'setTask', name: skillName, category: 'skill' };
        }
        // Property Actions (Sorted)
        const propNames = itemCategories["Properties"].sort();
        for (const propName of propNames) {
            actionMap[index++] = { type: 'setProperty', name: propName };
        }
        // Misc Actions (Sorted)
        const miscNames = itemCategories["Misc"].sort();
        for (const miscName of miscNames) {
            actionMap[index++] = { type: 'setMisc', name: miscName };
        }
        // Rebirth Actions
        actionMap[index++] = { type: 'rebirthOne' };
        actionMap[index++] = { type: 'rebirthTwo' };

        if (index !== actionSize) {
             console.warn(`Action map size (${index}) does not match calculated action size (${actionSize}). Check calculations.`);
             // Adjust actionSize if map is correct but calculation was off
             actionSize = index;
        }
         console.log("Action map built:", actionMap);

    } catch (e) {
        console.error("Error building action map. Is game data loaded?", e);
        actionMap = []; // Clear map on error
    }
}

// --- Agent Interaction Loop ---
let stepsSinceLastAction = 0;
const agentActionFrequency = 25; // Agent acts every N game ticks
const agentBatchSize = 32; // How many memories to replay each training step
let currentStateSnapshot = null; // Store snapshot for reward calculation

function agentStep() {
    if (!agent || isAgentPaused || typeof gameData === 'undefined') {
        // Don't run agent if not initialized, paused, or game not ready
        return;
    }

    // Use agentParams for frequency
    if (stepsSinceLastAction % agentParams.agentActionFrequency === 0) { 
        // 1. Get previous state snapshot (for reward calculation)
        const previousStateSnapshot = currentStateSnapshot || captureStateSnapshot(); // Use previous or capture fresh

        // 2. Get current state representation and available actions
        const currentState = getAgentState();
        const availableActions = getAvailableActionsMask();

        // 3. Agent chooses an action
        const actionIndex = agent.act(currentState, availableActions);

        // 4. Perform the action in the game
        performAgentAction(actionIndex);

        // --- Training Step --- 

        // 5. Capture the state *after* the action 
        const nextStateSnapshot = captureStateSnapshot(); 
        const nextState = getAgentState();
        const done = !nextStateSnapshot.isAlive; // Episode ends if dead

        // 6. Calculate reward
        const reward = calculateReward(previousStateSnapshot, nextStateSnapshot);

        // 7. Remember the transition
        agent.remember(currentState, actionIndex, reward, nextState, done);

        // 8. Train the agent (replay memory)
        agent.replay(agentParams.agentBatchSize); // Asynchronous, happens in background

        // Update current snapshot for the next step's reward calculation
        currentStateSnapshot = nextStateSnapshot;

        // Reset/Rebirth if the episode ended (player died)
        if (done) {
            console.log("Episode finished (death). Checking for Rebirth One availability...");
            currentStateSnapshot = null; // Reset snapshot for new life
            lastReward = 0; // Reset last reward display
            
            // Automatically trigger the first rebirth upon death, IF AVAILABLE
            if (typeof rebirthOne === 'function' && typeof getAvailableActionsMask === 'function') {
                const rebirthOneActionIndex = actionMap.findIndex(a => a && a.type === 'rebirthOne');
                if (rebirthOneActionIndex !== -1) {
                    const mask = getAvailableActionsMask();
                    if (mask[rebirthOneActionIndex]) {
                         console.log("Rebirth One is available. Calling rebirthOne().");
                         rebirthOne(); // Perform the rebirth
                         autoRebirthCounter++; // Increment the counter
                    } else {
                         console.warn("Agent died, but Rebirth One is not available according to action mask. Cannot auto-evolve yet.");
                         // Agent needs to learn to meet requirements first or choose rebirth manually
                    }
                } else {
                    console.error("Could not find rebirthOne action in actionMap.");
                }
            } else {
                 console.error("rebirthOne or getAvailableActionsMask function not found for automatic evolution.");
            }
        }
    }

    stepsSinceLastAction++;
}

// --- UI Control Functions ---

function pauseAgent() {
    if (!agent) return;
    console.log("Pausing Agent...");
    isAgentPaused = true;
    updateAgentControlButtonStates();
}

function resumeAgent() {
    if (!agent) return;
    console.log("Resuming Agent...");
    isAgentPaused = false;
    updateAgentControlButtonStates();
}

async function manualSaveAgent() {
    if (!agent) {
        console.error("Agent not initialized, cannot save.");
        return;
    }
    await agent.save();
    alert("Agent model saved to local storage."); // Simple feedback
}

async function manualLoadAgent() {
    if (!agent) {
        console.error("Agent not initialized, cannot load.");
        return;
    }
    await agent.load();
    // Reset epsilon decay potentially if loading a trained model
    // agent.epsilon = agent.epsilonMin; 
    alert("Attempted to load agent model from local storage."); // Simple feedback
}

function updateAgentControlButtonStates() {
    const pauseButton = document.getElementById('aiPauseButton');
    const resumeButton = document.getElementById('aiResumeButton');
    if (pauseButton && resumeButton) {
        pauseButton.disabled = isAgentPaused;
        resumeButton.disabled = !isAgentPaused;
    }
}

// Function to update the elements in the AI Agent Tab
function updateAgentTabUI() {
    if (!agent) return; 

    const statusEl = document.getElementById('aiStatus');
    const epsilonEl = document.getElementById('aiEpsilon');
    const memoryEl = document.getElementById('aiMemorySize');
    const rewardEl = document.getElementById('aiLastReward');
    const rebirthCounterEl = document.getElementById('aiAutoRebirths');
    // Max memory display is now handled by updateParamsUI
    // const maxMemoryEl = document.getElementById('aiMaxMemorySizeDisplay');

    if (statusEl) statusEl.textContent = isAgentPaused ? "Paused" : "Active";
    if (epsilonEl) epsilonEl.textContent = agent.epsilon.toFixed(4); // Display current epsilon
    if (memoryEl) memoryEl.textContent = agent.memory.length;
    // if (maxMemoryEl) maxMemoryEl.textContent = agentParams.maxMemorySize; // Handled by updateParamsUI
    if (rewardEl) rewardEl.textContent = lastReward.toFixed(3);
    if (rebirthCounterEl) rebirthCounterEl.textContent = autoRebirthCounter; 
    updateDemoCountUI(); // Update demo count frequently
}

// --- Periodic Saving ---
setInterval(() => {
    if (agent && !isAgentPaused) { // Only save if agent exists and is not paused
        agent.save().catch(err => console.error("Periodic save failed:", err));
    }
}, 60000); // Save every 60 seconds 

// --- Demonstration Storage Functions ---
function saveDemonstrations() {
    try {
        localStorage.setItem(DEMOS_STORAGE_KEY, JSON.stringify(demonstrations));
        console.log(`Saved ${demonstrations.length} demonstrations to localStorage.`);
        alert(`Saved ${demonstrations.length} demonstrations.`);
    } catch (e) {
        console.error("Error saving demonstrations:", e);
        alert("Error saving demonstrations. Check console.");
    }
}

function loadDemonstrations() {
    try {
        const savedDemos = localStorage.getItem(DEMOS_STORAGE_KEY);
        if (savedDemos) {
            demonstrations = JSON.parse(savedDemos);
            console.log(`Loaded ${demonstrations.length} demonstrations from localStorage.`);
            alert(`Loaded ${demonstrations.length} demonstrations.`);
        } else {
            demonstrations = [];
            console.log("No saved demonstrations found.");
        }
        updateDemoCountUI(); // Update UI display
    } catch (e) {
        console.error("Error loading demonstrations:", e);
        demonstrations = []; // Reset on error
        updateDemoCountUI();
        alert("Error loading demonstrations. Check console.");
    }
}

function clearDemonstrations() {
    if (confirm("Are you sure you want to clear all stored demonstrations? This cannot be undone.")) {
        demonstrations = [];
        localStorage.removeItem(DEMOS_STORAGE_KEY);
        console.log("Cleared demonstrations.");
        updateDemoCountUI();
    }
}

// --- UI Handler Functions for Demonstrations ---
function toggleRecording(isChecked) {
    isRecording = isChecked;
    console.log(`Demonstration recording ${isRecording ? 'ENABLED' : 'DISABLED'}`);
    // Optional: Add visual feedback to the UI
}

function updateDemoCountUI() {
    const demoCountEl = document.getElementById('demoCount');
    if (demoCountEl) {
        demoCountEl.textContent = demonstrations.length;
    }
}

// Functions called by buttons
function uiClearDemos() { clearDemonstrations(); }
function uiSaveDemos() { saveDemonstrations(); }
function uiLoadDemos() { loadDemonstrations(); }

function uiPretrainAgent() {
    if (!agent) {
        alert("Agent is not initialized yet.");
        return;
    }
    if (demonstrations.length === 0) {
        alert("No demonstrations loaded. Please load or record demonstrations first.");
        return;
    }
    
    // Pause the agent before starting pretraining
    if (typeof pauseAgent === 'function') {
        console.log("Pausing agent for pretraining...");
        pauseAgent();
    } else {
        console.warn("pauseAgent function not found, cannot pause automatically for pretraining.");
    }

    // Call the agent's pretrain method
    agent.pretrain(demonstrations).catch(err => {
         // Error handling improved in pretrain method itself
         // console.error("Pretraining failed:", err);
         // alert("Pretraining failed. Check console.");
    });
} 