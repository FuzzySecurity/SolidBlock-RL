const tf = require('@tensorflow/tfjs-node');
const { createModel } = require('./model');
const Game = require('../game/game.js');
const fs = require('fs');
const path = require('path');

// Define available actions.
const actions = ['up', 'down', 'left', 'right'];
const numActions = actions.length;

// For the expanded view: a 9x9 grid where each cell is one–hot encoded across 6 classes.
const gridSize = 9;
const numClasses = 6;

// Define epsilon parameters for training
let epsilon = 1.0;
const epsilonDecay = 0.999;  // The factor by which epsilon decays each update.
const epsilonMin = 0.1;      // The minimum threshold for epsilon.
const epsilonBoost = 0.2;    // The amount to boost epsilon when it decays below epsilonMin.

// Periodic exploration boost constants
const explorationCycleLength = 1200;  // Every 1200 episodes, boost ε instead of decaying normally.
const epsilonResetCycle = 0.2;        // Maximum reset boost for ε

// Checkpoint helpers

/**
 * Save the current training state to a JSON file.
 * @param {Object} state - The training state to save.
 * @param {string} [checkpointPath='checkpoint.json'] - The file path to save the checkpoint.
 */
function saveTrainingState(state, checkpointPath = 'checkpoint.json') {
  try {
    fs.writeFileSync(checkpointPath, JSON.stringify(state), 'utf8');
    console.log(`Training state saved to ${checkpointPath}`);
  } catch (err) {
    console.error("Error saving training state:", err);
  }
}

/**
 * Load the training state from a JSON file.
 * @param {string} checkpointPath - The checkpoint file path.
 * @returns {Object|null} The training state if found, otherwise null.
 */
function loadTrainingState(checkpointPath = 'checkpoint.json') {
  try {
    if (fs.existsSync(checkpointPath)) {
      const data = fs.readFileSync(checkpointPath, 'utf8');
      const state = JSON.parse(data);
      console.log(`Training state loaded from ${checkpointPath}`);
      return state;
    } else {
      console.log("No checkpoint file found; starting fresh.");
      return null;
    }
  } catch (err) {
    console.error("Error loading training state:", err);
    return null;
  }
}

// Feature extraction helpers

/**
 * Determine the cell's category in the game state.
 * Categories:
 *   0: Out-of-bounds or an obstacle.
 *   1: Red bonus block.
 *   2: Green bonus block.
 *   3: Food.
 *   4: Player.
 *   5: Empty cell.
 * @param {number} x - The x-coordinate of the cell.
 * @param {number} y - The y-coordinate of the cell.
 * @param {Object} state - The current game state.
 * @param {Object} game - The game instance (providing board dimensions).
 * @returns {number} The category index.
 */
function determineCategory(x, y, state, game) {
  if (x < 0 || x >= game.width || y < 0 || y >= game.height) return 0;
  if (state.obstacles.some(o => o.x === x && o.y === y)) return 0;
  if (state.redBlocks.some(rb => rb.x === x && rb.y === y)) return 1;
  if (state.greenBlocks.some(gb => gb.x === x && gb.y === y)) return 2;
  if (state.food && state.food.x === x && state.food.y === y) return 3;
  if (state.player.x === x && state.player.y === y) return 4;
  return 5;
}

/**
 * Convert a category number into a one-hot encoded array.
 * @param {number} category - The category index.
 * @param {number} [numCategories=numClasses] - The total number of categories.
 * @returns {number[]} The one-hot encoded vector.
 */
function oneHotEncode(category, numCategories = numClasses) {
  const vec = new Array(numCategories).fill(0);
  vec[category] = 1;
  return vec;
}

/**
 * Generate the feature representation from the game state.
 * This creates a 9x9 grid (one-hot encoded per cell) and computes a normalized offset
 * between the player and the food.
 * @param {Object} state - The current game state.
 * @param {Object} game - The game instance.
 * @returns {{grid: number[][][], offset: number[]}} The feature vector.
 */
function getFeatureVector(state, game) {
  const player = state.player;
  const food = state.food;
  const dx = 5 * (food.x - player.x) / game.width;
  const dy = 5 * (food.y - player.y) / game.height;
  
  const grid = [];
  const half = Math.floor(gridSize / 2); // half = 4 for gridSize=9.
  for (let i = -half; i <= half; i++) {
    const row = [];
    for (let j = -half; j <= half; j++) {
      const cellX = player.x + j;
      const cellY = player.y + i;
      const cat = determineCategory(cellX, cellY, state, game);
      row.push(oneHotEncode(cat, numClasses));
    }
    grid.push(row);
  }
  return { grid: grid, offset: [dx, dy] };
}

// Prioritized replay sampling

/**
 * Sample a batch of transitions using a mix of uniform and prioritized replay.
 * @param {Object[]} buffer - The replay buffer containing transitions.
 * @param {number} batchSize - The desired number of transitions.
 * @returns {Object[]} The sampled batch of transitions.
 */
function sampleEnhancedBatch(buffer, batchSize) {
  // If there are fewer transitions than requested, return a shallow copy.
  if (buffer.length < batchSize) {
    return buffer.slice();
  }
  
  // We will return half uniform samples and half prioritized samples.
  const halfBatch = Math.floor(batchSize / 2);
  const uniformSamples = [];
  
  // Uniform sampling: select halfBatch samples uniformly at random.
  for (let i = 0; i < halfBatch; i++) {
    const index = Math.floor(Math.random() * buffer.length);
    uniformSamples.push(buffer[index]);
  }
  
  // Now compute effective priorities with an exponential recency bias.
  // For each transition, compute a recency weight based on its index.
  // Newer transitions (with higher index values) receive a weight closer to 1.
  // Adjust the exponent alpha to tune how much more important recency is.
  const alpha = 2.0;
  const effectivePriorities = [];
  
  // Assume that the replay buffer is an array where the oldest entry is at index 0
  // and the newest entry is at index buffer.length - 1.
  for (let i = 0; i < buffer.length; i++) {
    const transition = buffer[i];
    const originalPriority = (transition.priority !== undefined) 
                             ? transition.priority 
                             : Math.abs(transition.reward);
    // Recency weight: newest transitions get weight 1, oldest get weight ~ (1 / buffer.length)
    const recencyWeight = ((i + 1) / buffer.length);
    // Effective priority is the product of the original priority and the recency weight raised to alpha.
    const effectivePriority = originalPriority * Math.pow(recencyWeight, alpha);
    effectivePriorities.push(effectivePriority);
  }
  
  // Sum up the effective priorities.
  const totalPriority = effectivePriorities.reduce((sum, p) => sum + p, 0);
  
  const prioritizedSamples = [];
  // For halfBatch samples, select transitions proportional to their effective priority.
  for (let i = 0; i < halfBatch; i++) {
    let random = Math.random();
    let cumulative = 0;
    for (let j = 0; j < effectivePriorities.length; j++) {
      cumulative += effectivePriorities[j] / totalPriority;
      if (random < cumulative) {
        prioritizedSamples.push(buffer[j]);
        break;
      }
    }
  }
  
  // Combine the uniform and prioritized samples.
  return uniformSamples.concat(prioritizedSamples);
}

/**
 * Randomly sample a batch of transitions from the replay buffer.
 * @param {Array} buffer - The replay buffer.
 * @param {number} batchSize - The number of transitions to sample.
 * @returns {Array} The sampled transitions.
 */
function sampleRandomBatch(buffer, batchSize) {
  const sample = [];
  for (let i = 0; i < batchSize; i++) {
    const index = Math.floor(Math.random() * buffer.length);
    sample.push(buffer[index]);
  }
  return sample;
}

// Train model with checkpointing
/**
 * Train the model using experiences stored in a replay buffer.
 * The training process features checkpointing and listens for user input to stop or save snapshots.
 * @param {string|null} [resumeModelPath=null] - Path to resume training from a saved checkpoint.
 */
async function train(resumeModelPath = null) {
  let model;
  let trainingState = null;
  const checkpointPath = 'checkpoint.json';

  if (resumeModelPath) {
    const fullPath = 'file://' + resumeModelPath;
    console.log("Loading existing model from:", fullPath);
    try {
      model = await tf.loadLayersModel(fullPath);
      console.log("Model loaded successfully. Resuming training...");
    } catch (err) {
      console.error("Failed to load model. Creating a fresh model instead. Error:", err);
      model = createModel(gridSize, numClasses, numActions);
    }
    // Load saved training state if it exists.
    trainingState = loadTrainingState(checkpointPath);
  } else {
    model = createModel(gridSize, numClasses, numActions);
  }

  // Hyperparameters.
  const maxSteps = 500;
  const GAMMA = 0.99;
  const TAU = 0.005;
  const REWARD_MIN = -20;
  const REWARD_MAX = 20;
  const STEP_PENALTY = 0.05;
  const REPLAY_BUFFER_SIZE = 10000;
  const BATCH_SIZE = 32;
  const CLIP_VALUE = 5;
  const USE_PRIORITIZED_REPLAY = true;
  const HARD_UPDATE_FREQUENCY = 20;

  let baseLearningRate = 0.001;
  const learningRateDecayFactor = 0.98;
  const optimizerUpdateInterval = 200;
  let currentOptimizer = tf.train.adam(baseLearningRate);
  let optimizer = currentOptimizer;

  // Setup stop signal.
  let stopTraining = false;
  console.log("Training started. Press ENTER to stop training and save the model. Press 's' to save a snapshot and training state.");
  process.stdin.setEncoding('utf8');
  process.stdin.setRawMode(true);
  process.stdin.on("data", (key) => {
    if (key === '\r' || key === '\n') {
      console.log("\nStop signal received. Finishing current episode and saving model...");
      stopTraining = true;
    } else if (key.toLowerCase() === 's') {
      console.log("\nSave signal received. Saving current model and training state...");
      const dateTimeString = new Date().toISOString().replace(/[:.]/g, '-');
      const snapshotDir = `./model_snapshot_${dateTimeString}`;
      model.save(`file://${snapshotDir}`)
        .then(() => {
          const stateToSave = {
            episode,
            epsilon,
            replayBuffer,
            episodeLosses,
            episodeMaxQs,
            baseLearningRate
          };
          saveTrainingState(stateToSave, path.join(snapshotDir, 'checkpoint.json'));
          console.log("\nModel and training state saved.");
        })
        .catch(err => {
          console.error("Model save error:", err);
        });
    }
  });

  // Create target network.
  const targetModel = createModel(gridSize, numClasses, numActions);
  targetModel.setWeights(model.getWeights());

  // Default training state variables.
  let episode = 1;
  let replayBuffer = [];
  let episodeLosses = [];
  let episodeMaxQs = [];
  const episodeStats = [];
  let latestLoss = 0;

  // Restore training state if loaded.
  if (trainingState) {
    episode = trainingState.episode;
    epsilon = trainingState.epsilon;
    replayBuffer = trainingState.replayBuffer || [];
    episodeLosses = trainingState.episodeLosses || [];
    episodeMaxQs = trainingState.episodeMaxQs || [];
    baseLearningRate = trainingState.baseLearningRate || baseLearningRate;
    console.log(`Resumed at episode ${episode}, epsilon=${epsilon}`);
  }

  // Ensure epsilon is at least 1.0 for a fresh start if desired.
  epsilon = Math.max(epsilon, 1.0);

  while (true) {
    let totalReward = 0;
    const game = new Game();
    let currentState = game.getState();
    let currentFeature = getFeatureVector(currentState, game);

    let step;
    for (step = 0; step < maxSteps; step++) {
      let actionIndex;
      if (Math.random() < epsilon) {
        actionIndex = Math.floor(Math.random() * numActions);
      } else {
        const gridTensor = tf.tensor4d([currentFeature.grid], [1, gridSize, gridSize, numClasses]);
        const offsetTensor = tf.tensor2d([currentFeature.offset], [1, 2]);
        const qsTensor = model.predict([gridTensor, offsetTensor]);
        const qs = qsTensor.dataSync();
        episodeMaxQs.push(Math.max(...qs));
        qsTensor.dispose();
        gridTensor.dispose();
        offsetTensor.dispose();
        actionIndex = qs.indexOf(Math.max(...qs));
      }
      const action = actions[actionIndex];
      const { state: nextState, reward, done } = game.step(action);

      let rawReward = reward;
      let penalizedReward = rawReward - STEP_PENALTY;
      let clippedReward = Math.max(REWARD_MIN, Math.min(REWARD_MAX, penalizedReward));
      totalReward += clippedReward;

      if (done) {
        break;
      }

      const nextFeature = getFeatureVector(nextState, game);

      replayBuffer.push({
        state: currentFeature,
        action: actionIndex,
        reward: clippedReward,
        nextState: nextFeature,
        done: done,
        priority: Math.abs(clippedReward)
      });

      // Remove extra oldest transitions if needed.
      if (replayBuffer.length > REPLAY_BUFFER_SIZE) {
        replayBuffer.splice(0, replayBuffer.length - REPLAY_BUFFER_SIZE);
      }

      if (replayBuffer.length >= BATCH_SIZE) {
        const batch = USE_PRIORITIZED_REPLAY
          ? sampleEnhancedBatch(replayBuffer, BATCH_SIZE)
          : sampleRandomBatch(replayBuffer, BATCH_SIZE);

        const batchLoss = await tf.tidy(() => {
          const gridBatch = batch.map(t => t.state.grid);
          const offsetBatch = batch.map(t => t.state.offset);
          const nextGridBatch = batch.map(t => t.nextState.grid);
          const nextOffsetBatch = batch.map(t => t.nextState.offset);
          const rewardsBatch = tf.tensor1d(batch.map(t => t.reward));
          const donesBatch = tf.tensor1d(batch.map(t => t.done ? 1 : 0), 'float32');

          const gridTensor = tf.tensor4d(gridBatch, [BATCH_SIZE, gridSize, gridSize, numClasses]);
          const offsetTensor = tf.tensor2d(offsetBatch, [BATCH_SIZE, 2]);
          const nextGridTensor = tf.tensor4d(nextGridBatch, [BATCH_SIZE, gridSize, gridSize, numClasses]);
          const nextOffsetTensor = tf.tensor2d(nextOffsetBatch, [BATCH_SIZE, 2]);

          const nextQsMain = model.predict([nextGridTensor, nextOffsetTensor]);
          const bestActionsTensor = nextQsMain.argMax(1);
          const nextQsTarget = targetModel.predict([nextGridTensor, nextOffsetTensor]);
          const bestActionsOneHot = tf.oneHot(bestActionsTensor, numActions);
          const bestQTargetTensor = nextQsTarget.mul(bestActionsOneHot).sum(1);
          const oneTensor = tf.scalar(1);
          const targets = rewardsBatch.add(
            bestQTargetTensor.mul(GAMMA).mul(oneTensor.sub(donesBatch))
          );

          const qsPred = model.predict([gridTensor, offsetTensor]);
          const actionsTensor1d = tf.tensor1d(batch.map(t => t.action), 'int32');
          const actionOneHot = tf.oneHot(actionsTensor1d, numActions);
          const predictedTaken = qsPred.mul(actionOneHot).sum(1);

          const tdErrorTensor = tf.abs(targets.sub(predictedTaken));
          const tdErrorArray = tdErrorTensor.dataSync();
          for (let i = 0; i < batch.length; i++) {
            batch[i].priority = tdErrorArray[i];
          }

          const loss = tf.losses.huberLoss(targets, predictedTaken);

          const gradsAndValue = tf.variableGrads(() => {
            const qsPredInner = model.predict([gridTensor, offsetTensor]);
            const predictedTakenInner = qsPredInner.mul(actionOneHot).sum(1);
            return tf.losses.huberLoss(targets, predictedTakenInner);
          });
          for (const key in gradsAndValue.grads) {
            gradsAndValue.grads[key] = tf.clipByValue(gradsAndValue.grads[key], -CLIP_VALUE, CLIP_VALUE);
          }
          optimizer.applyGradients(gradsAndValue.grads);
          return gradsAndValue.value.dataSync()[0];
        });
        latestLoss = batchLoss;
        episodeLosses.push(batchLoss);
      }

      currentFeature = nextFeature;

      if (step % 50 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      if (done) {
        break;
      }
    } // end step loop

    console.log(`Episode ${episode} complete: Steps = ${step}, Total Reward = ${totalReward.toFixed(2)}`);

    const meanLoss = episodeLosses.length > 0
      ? (episodeLosses.reduce((a, b) => a + b, 0) / episodeLosses.length)
      : 0;
    const sortedLosses = episodeLosses.slice().sort((a, b) => a - b);
    const medianLoss = (sortedLosses.length % 2 === 0)
      ? (sortedLosses[sortedLosses.length / 2 - 1] + sortedLosses[sortedLosses.length / 2]) / 2
      : sortedLosses[Math.floor(sortedLosses.length / 2)];

    const avgMaxQ = episodeMaxQs.length > 0
      ? (episodeMaxQs.reduce((a, b) => a + b, 0) / episodeMaxQs.length)
      : 0;

    episodeStats.push({ episode, totalReward, meanLoss, medianLoss, avgMaxQ });

    if (episode % HARD_UPDATE_FREQUENCY === 0) {
      console.clear();
      const recent = episodeStats.slice(-20);
      const avgReward = recent.reduce((sum, stat) => sum + stat.totalReward, 0) / recent.length;
      const avgMeanLoss = recent.reduce((sum, stat) => sum + stat.meanLoss, 0) / recent.length;
      const avgMedianLoss = recent.reduce((sum, stat) => sum + stat.medianLoss, 0) / recent.length;
      const avgQ = recent.reduce((sum, stat) => sum + stat.avgMaxQ, 0) / recent.length;
      console.log(`\nEpisodes ${episode - 19}-${episode} summary:`);
      console.log(`  Avg Reward      = ${avgReward.toFixed(2)}`);
      console.log(`  Avg Mean Loss   = ${avgMeanLoss.toFixed(5)}`);
      console.log(`  Avg Median Loss = ${avgMedianLoss.toFixed(5)}`);
      console.log(`  Avg Max Q Value = ${avgQ.toFixed(5)}`);
      console.log(`  Epsilon         = ${epsilon.toFixed(5)}\n`);
    }

    episode++;

    if (episode % optimizerUpdateInterval === 0) {
      const epochsPassed = Math.floor(episode / optimizerUpdateInterval);
      const newLR = baseLearningRate * Math.pow(learningRateDecayFactor, epochsPassed);
      console.log(`Updating optimizer: new learning rate = ${newLR.toFixed(6)}`);
      optimizer = tf.train.adam(newLR);
    }

    if (episode % explorationCycleLength === 0) {
      // Every explorationCycleLength, boost epsilon by a random amount
      // between epsilonMin (0.1) and epsilonResetCycle (0.2)
      const boostAmount = epsilonMin + Math.random() * (epsilonResetCycle - epsilonMin);
      epsilon = epsilon + boostAmount;
    } else {
      // Otherwise, decay epsilon multiplicatively
      epsilon = epsilon * epsilonDecay;
      // Soft reset: if epsilon falls below the minimum threshold, reset it to epsilonBoost (0.2)
      if (epsilon < epsilonMin) {
        epsilon = epsilonBoost;
      }
    }

    if (episode % 500 === 0 && replayBuffer.length > 0) {
      console.log("Purging oldest 20% of replay buffer to refresh data...");
      const numToRemove = Math.floor(replayBuffer.length * 0.2);
      replayBuffer.splice(0, numToRemove);
    }

    // Soft update of target network.
    const mainWeights = model.getWeights();
    const targetWeights = targetModel.getWeights();
    const updatedTargetWeights = mainWeights.map((mainW, i) =>
      tf.add(tf.mul(mainW, TAU), tf.mul(targetWeights[i], 1 - TAU))
    );
    targetModel.setWeights(updatedTargetWeights);

    if (episode % HARD_UPDATE_FREQUENCY === 0) {
      console.log(`Performed soft update for target network at episode ${episode}`);
    }

    // Auto-save a snapshot (and training state) every 200 episodes.
    if (episode % 200 === 0) {
      const snapshotRange = episodeStats.length >= 200 ? episodeStats.slice(-200) : episodeStats;
      const snapshotAvgReward = snapshotRange.reduce((sum, stat) => sum + stat.totalReward, 0) / snapshotRange.length;
      const snapshotRewardStr = snapshotAvgReward.toFixed(2);
      const snapshotDatetime = new Date().toISOString().replace(/[:.]/g, '-');
      const snapshotDir = `./snapshot_${episode}_${snapshotRewardStr}_${snapshotDatetime}`;
      model.save(`file://${snapshotDir}`)
        .then(() => {
          const stateToSave = {
            episode,
            epsilon,
            replayBuffer,
            episodeLosses,
            episodeMaxQs,
            baseLearningRate
          };
          saveTrainingState(stateToSave, path.join(snapshotDir, 'checkpoint.json'));
          console.log(`Auto-saved snapshot and training state: ${snapshotDir}`);
        })
        .catch(err => console.error("Auto-snapshot save error:", err));
    }

    await new Promise(resolve => setTimeout(resolve, 10));
    if (stopTraining) {
      break;
    }
  }

  console.log("Saving final model and training state to disk...");
  const dateTimeString = new Date().toISOString().replace(/[:.]/g, '-');
  const finalDir = `./saved-model_${dateTimeString}`;
  await model.save(`file://${finalDir}`);
  const finalState = {
    episode,
    epsilon,
    replayBuffer,
    episodeLosses,
    episodeMaxQs,
    baseLearningRate
  };
  saveTrainingState(finalState, path.join(finalDir, 'checkpoint.json'));
  console.log("Final model and training state saved. Training complete.");

  process.stdin.setRawMode(false);
  process.stdin.pause();
  process.exit(0);
}

module.exports = { train };
