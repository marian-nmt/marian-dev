# Layers

Typically, nodes are organised in layers in a deep neural network. 
Layers can be regarded as the highest-level building blocks in neural networks, which perform different kinds of transformations on their inputs. 
In Marian, a layer wraps a group of nodes, which performs a specific mathmatical manipulation. 
Layers can offer a shortcut for creating a complex model. 
Take `mlp::dense` layer for example. 
The `mlp::dense` layer represents a fully connected layer, 
which implements the operations `output = activation(input * weight + bias)`. 
We can construct a dense layer in the graph running the following code: 
```cpp
// add input node x
auto x = graph->constant({120,5}, inits::fromVector(inputData));
// construct a dense layer in the graph
auto layer1 = mlp::dense()                           
      ("prefix", "layer1")                  // prefix name is layer1
      ("dim", 5)                            // output dimension is 5
      ("activation", (int)mlp::act::tanh)   // activation function is tanh
      .construct(graph)->apply(x);          // construct this layer in graph
                                            // and link node x as the input
```
The options for this layer are passed by a pair of `(key, value)`,
where `key` is a predefined option, and `value` is the option parameter you select.
After the options, we call `construct()` to create a layer instance in the graph,
and `apply()` to link the input with this layer. 
You can also define the same layer by adding nodes and operations:
```cpp
// construct a dense layer using nodes
auto W1 = graph->param("W1", {120, 5}, inits::glorotUniform());
auto b1 = graph->param("b1", {1, 5}, inits::zeros());
auto h = tanh(affine(x, W1, b1));
```
There are four categories of layers implemented in Marian.

## convolution layer

To use `convlolution` layer, you first need to install [NVIDIA cuDNN](https://developer.nvidia.com/cudnn). The convolution layer supported by Marian is a 2D [convolution layer](https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layers). This layer  creates a convolution kernel which is used to convolved with the input. The options of `convolution` layer are listed beblow:

| Option Name   | Definition     | Value Type    | Default Value  |
| ------------- |----------------|---------------|---------------|
| prefix        | Prefix name (used to form the parameter names) | `std::string` | `None` |
| kernel-dims   | The height and width of the kernel | `std::pair<int, int>` | `None`| 
| kernel-num    | The number of kernel | `int` | `None`       |
| paddings      | The height and width of paddings | `std::pair<int, int>` | `(0,0)`|
| strides       | The height and width of strides | `std::pair<int, int>` | `(1,1)` |

Example:
```cpp
// construct a convolution layer
auto conv_1 = convolution(graph)              // pass graph pointer to the layer 
      ("prefix", "conv_1")                    // prefix name is conv_1
      ("kernel-dims", std::make_pair(3,3))    // kernel is 3*3
      ("kernel-num", 32)                      // kernel no. is 32
      .apply(x);                              // link node x as the input
```

## mlp layers

Marian offers `mlp::mlp` to create a [multilayer perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) network. 
The `mlp::mlp` is a container which can stack multiple layers using `push_back()` function. 
There are two types of mlp layers provided by Marian: `mlp::dense` and `mlp::ouput`.
The `mlp::dense` layer, as introduced before, is a fully connected layer. 
The options of `mlp::dense` layer are listed beblow:
 
| Option Name   | Definition     | Value Type    | Default Value  |
| ------------- |----------------|---------------|---------------|
| prefix        | Prefix name (used to form the parameter names) | `std::string` | `None` |
| dim           | Output dimension | `int` | `None` |
| layer-normalization | Whether to normalise the layer output or not | `bool` | `false` |
| nematus-normalization | Whether to use Nematus layer normalisation or not | `bool` | `false` |
| activation | Activation function | `int` | `act::linear` |

The `mlp::ouput` layer is used to construct an output layer. 
In the context of neural machine translation, you can tie embedding layers to `mlp::ouput` layer using `tieTransposed()`, or set shortlisted words using `setShortlist()`.
The options of `mlp::output` layer are listed beblow:
 
| Option Name   | Definition     | Value Type    | Default Value  |
| ------------- |----------------|---------------|---------------|
| prefix        | Prefix name (used to form the parameter names) | `std::string` | `None` |
| dim           | Output dimension | `int` | `None` |
| vocab         | File path to the factored vocabulary | `std::string` | `None` |
| output-omit-bias | Whether this layer has a bias parameter | `bool` | `true` |
| lemma-dim-emb | Re-embedding dimension of lemma in factors, must be used with `vocab` option | `int` | `0` |
| output-approx-knn | Parameters for LSH-based output approximation, i.e., `k` (the first element) and `nbit` (the second element) | `std::vector<int>` | None |

## rnn layers
It is under development.

## encoder & decoder layers
It is under development.
