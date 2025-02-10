const tf = require('@tensorflow/tfjs-node');

/**
 * Creates a convolutional neural network model that accepts two inputs:
 *  - A grid input of shape [gridSize, gridSize, numClasses] (the visual field).
 *    For a 9x9 view, set gridSize = 9.
 *  - An offset input of shape [2] (the normalized relative food offset).
 *
 * The model processes the grid input through two convolutional layers followed
 * by a max-pooling layer to reduce spatial dimensions. The resulting features are
 * then flattened and concatenated with the offset input. Three dense layers follow
 * before outputting Q–values (one per action).
 *
 * This architecture is designed to capture a broader field of view (9×9)
 * and provide increased capacity for long-term planning and obstacle avoidance.
 *
 * @param {number} gridSize - The height/width of the grid (e.g., 9 for a 9x9 view).
 * @param {number} numClasses - The number of classes for one–hot encoding (e.g., 6).
 * @param {number} outputDim - The number of actions (output neurons).
 * @returns {tf.Model} A compiled TensorFlow.js model.
 */
function createModel(gridSize, numClasses, outputDim) {
  const gridInput = tf.input({ shape: [gridSize, gridSize, numClasses] });
  const offsetInput = tf.input({ shape: [2] });

  // Convolutional layers for processing the grid
  let x = tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }).apply(gridInput);

  x = tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }).apply(x);

  // Max-pooling to reduce spatial dimensions
  x = tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2],
    padding: 'same'
  }).apply(x);

  // Flatten the convolutional output
  x = tf.layers.flatten().apply(x);

  // Concatenate with the offset input
  const concatenated = tf.layers.concatenate().apply([x, offsetInput]);

  // Dense layers
  let dense = tf.layers.dense({ units: 256, activation: 'relu' }).apply(concatenated);
  dense = tf.layers.dropout({ rate: 0.2 }).apply(dense);
  dense = tf.layers.dense({ units: 128, activation: 'relu' }).apply(dense);
  dense = tf.layers.dense({ units: 64, activation: 'relu' }).apply(dense);

  // Output layer
  const output = tf.layers.dense({ units: outputDim, activation: 'linear' }).apply(dense);

  const model = tf.model({ inputs: [gridInput, offsetInput], outputs: output });
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError'
  });

  return model;
}

module.exports = { createModel };
