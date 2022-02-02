// Other techniques for learning

class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let sigmoid = new ActivationFunction(
  x => 1 / (1 + Math.exp(-x)),
  y => y * (1 - y)
);

let tanh = new ActivationFunction(
  x => Math.tanh(x),
  y => 1 - (y * y)
);


class NeuralNetwork {
  /*
  * if first argument is a NeuralNetwork the constructor clones it
  * USAGE: cloned_nn = new NeuralNetwork(to_clone_nn);
  */
  constructor(in_nodes, hid_nodes, out_nodes) {
    if (in_nodes instanceof NeuralNetwork) {
      let a = in_nodes;
      this.input_nodes = a.input_nodes;
      this.hidden_nodes = a.hidden_nodes;
      this.output_nodes = a.output_nodes;

      this.weights_ih = math.clone(a.weights_ih);
      this.weights_ho = math.clone(a.weights_ho);

      this.bias_h = math.clone(a.bias_h);
      this.bias_o = math.clone(a.bias_o);
    } else {
      this.input_nodes = in_nodes;
      this.hidden_nodes = hid_nodes;
      this.output_nodes = out_nodes;

      this.weights_ih = math.random(this.hidden_nodes, this.input_nodes);
      this.weights_ho = math.random(this.output_nodes, this.hidden_nodes);

      this.bias_h = math.random(this.hidden_nodes);
      this.bias_o = math.random(this.output_nodes);
    }

    // TODO: copy these as well
    this.setLearningRate();
    this.setActivationFunction();


  }

  predict(input_array) {

    // Generating the Hidden Outputs
    let inputs = math.matrix(input_array);
    let hidden = math.multiply(this.weights_ih, inputs);
    hidden = math.add(hidden, this.bias_h);
    // activation function!
    hidden = hidden.map(this.activation_function.func);

    // Generating the output's output!
    let output = math.multiply(this.weights_ho, hidden);
    output = math.add(output, this.bias_o);
    output = output.map(this.activation_function.func);

    // Sending back to the caller!
    return output.toArray();
  }

  setLearningRate(learning_rate = 0.1) {
    this.learning_rate = learning_rate;
  }

  setActivationFunction(func = sigmoid) {
    this.activation_function = func;
  }

  train(input_array, target_array) {
    // Generating the Hidden Outputs
    let inputs = math.matrix(input_array);
    let hidden = math.multiply(this.weights_ih, inputs);
    hidden = math.add(hidden, this.bias_h);
    // activation function!
    hidden = hidden.map(this.activation_function.func);

    // Generating the output's output!
    let outputs = math.multiply(this.weights_ho, hidden);
    outputs = math.add(outputs, this.bias_o);
    outputs = outputs.map(this.activation_function.func);

    // Convert array to matrix object
    let targets = math.matrix(target_array);

    // Calculate the error
    // ERROR = TARGETS - OUTPUTS
    let output_errors = math.subtract(targets, outputs);

    // let gradient = outputs * (1 - outputs);
    // Calculate gradient
    let gradients = math.map(outputs, this.activation_function.dfunc);
    gradients = math.multiply(gradients, output_errors);
    gradients = math.multiply(gradients, this.learning_rate);


    // Calculate deltas
    let hidden_T = math.transpose(hidden);
    let weight_ho_deltas = math.multiply(gradients, hidden_T);

    // Adjust the weights by deltas
    this.weights_ho = math.add(this.weights_ho, weight_ho_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_o = math.add(this.bias_o, gradients);

    // Calculate the hidden layer errors
    let who_t = math.transpose(this.weights_ho);
    let hidden_errors = math.multiply(who_t, output_errors);

    // Calculate hidden gradient
    let hidden_gradient = math.map(hidden, this.activation_function.dfunc);
    hidden_gradient = math.multiply(hidden_gradient, hidden_errors);
    hidden_gradient = math.multiply(hidden_gradient, this.learning_rate);

    // Calcuate input->hidden deltas
    let inputs_T = math.transpose(inputs);
    let weight_ih_deltas = math.multiply(hidden_gradient, inputs_T);

    this.weights_ih = math.add(this.weights_ih, weight_ih_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_h = math.add(this.bias_h, hidden_gradient);

    // outputs.print();
    // targets.print();
    // error.print();
  }

  serialize() {
    return JSON.stringify(this);
  }

  // static deserialize(data) {
  //   if (typeof data == 'string') {
  //     data = JSON.parse(data);
  //   }
  //   let nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes, data.output_nodes);
  //   nn.weights_ih = Matrix.deserialize(data.weights_ih);
  //   nn.weights_ho = Matrix.deserialize(data.weights_ho);
  //   nn.bias_h = Matrix.deserialize(data.bias_h);
  //   nn.bias_o = Matrix.deserialize(data.bias_o);
  //   nn.learning_rate = data.learning_rate;
  //   return nn;
  // }


  // Adding function for neuro-evolution
  copy() {
    return new NeuralNetwork(this);
  }

  // Accept an arbitrary function for mutation
  mutate(func) {
    this.weights_ih = func(this.weights_ih)
    this.weights_ho = func(this.weights_ho)
    this.bias_h = func(this.bias_h)
    this.bias_o = func(this.bias_o)
  }



}
