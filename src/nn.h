#ifndef NN_H
#define NN_H
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "engine.h"

typedef Value *(*activation)(struct Value *);

typedef struct Neuron
{
  size_t num_inputs;
  struct Value **weights;
  struct Value *bias;
  activation activation_fun;
} Neuron;

typedef struct Layer
{
  size_t num_neurons;
  struct Neuron **neurons;
} Layer;

typedef struct Network
{
  size_t num_inputs;
  size_t num_outputs;
  size_t num_layers;
  struct Layer **layers;
} Network;

// Construction
Neuron *initNeuron(size_t num_inputs, activation activation_fun);
Layer *creatLayer(size_t num_inputs, size_t num_neurons,activation activation_fun);
Network *createNetwork(size_t num_inputs, size_t num_outputs, size_t num_layers, size_t *layers, activation *activations);
// Computation
Value **forward(Network *net, Value **inpt);
// Cost
Value *mse(Value *logit, Value *target);
Value *crossEntropy(Value *logit, Value *target);
// Optimizers
void sgd(Network *net, double lr, bool bias);
void zeroGrad(Network *net);

void freeNeuron(struct Neuron *neuron);
void freeNet(struct Network *net);
void freeLayer(struct Layer *layer);

#endif

// #ifdef NN_IMPLEMENTATION

Value **forward(Network *net, Value **inpt)
{
  Value **activations = inpt;

  for (size_t layer_id = 0; layer_id < net->num_layers; ++layer_id)
  {
    Layer *layer = net->layers[layer_id];

    Value **logits = (Value **)calloc(layer->num_neurons, sizeof(Value *));

    for (size_t neuron_id = 0; neuron_id < layer->num_neurons; ++neuron_id)
    {
      Value *logit = initValue(0); // let's allocate on the stack
      Neuron *neuron = layer->neurons[neuron_id];

      for (size_t i = 0; i < neuron->num_inputs; ++i)
      {
        neuron->weights[i]->ref_count = 1.0; // reset ref_count

        logit = _add(logit, _mul(activations[i], neuron->weights[i]));
      }

      neuron->bias->ref_count = 1.0; // reset ref_count
      logit = _add(logit, neuron->bias);

      if(neuron->activation_fun == NULL)
        logit = logit;
      else
        logit = neuron->activation_fun(logit);

      logits[neuron_id] = logit;
    }
    // free(activations);
    activations = logits;
  }
  // for(size_t i = 0; i < net->num_outputs; ++i)
  //   activations[i]->type = OUTPUT;
  return activations;
}

Neuron *initNeuron(size_t num_inputs, activation activation_fun)
{
  Neuron *neuron = (Neuron *)calloc(1, sizeof(Neuron));

  double std_dev = sqrt(2.0 / num_inputs);

  if (neuron == NULL)
  {
    fprintf(stderr, "Error allocating memory for a neuron\n");
    return NULL;
  }
  neuron->num_inputs = num_inputs;
  neuron->weights = (Value **)calloc(num_inputs, sizeof(Neuron *));

  for (size_t i = 0; i < num_inputs; ++i)
  {
    // change seed everytime
    double data = (double)(rand() % RAND_MAX) / RAND_MAX * 2.0 * std_dev - std_dev;
    Value *w = initValue(data);
    neuron->weights[i] = w;
    neuron->activation_fun = activation_fun;
  }

  neuron->bias = initValue(0.0);
  return neuron;
}

Layer *createLayer(size_t num_inputs, size_t num_neurons, activation activation_fun)
{
  // creat a layer
  Layer *layer = (Layer *)calloc(1, sizeof(Layer));
  if (layer == NULL)
  {
    fprintf(stderr, "Error allocating memory for a layer\n");
    return NULL;
  }
  layer->num_neurons = num_neurons;
  layer->neurons = (Neuron **)calloc(num_neurons, sizeof(Neuron *));

  for (size_t i = 0; i < num_neurons; ++i)
  {
    Neuron *n = initNeuron(num_inputs, activation_fun);
    layer->neurons[i] = n;
  }
  return layer;
}

Network *createNetwork(size_t num_inputs, size_t num_outputs, size_t num_layers, size_t *layers, activation *activations)
{
  Network *net = (Network *)calloc(1, sizeof(Network));
  if (net == NULL)
  {
    fprintf(stderr, "Error allocating memory for a network\n");
    return NULL;
  }
  // initialize
  net->num_inputs = num_inputs;
  net->num_outputs = num_outputs;
  net->num_layers = num_layers;
  net->layers = (Layer **)calloc(num_layers, sizeof(Layer *));

  // Create layers
  size_t layer_input = num_inputs;
  for (size_t i = 0; i < num_layers; ++i)
  {
    Layer *layer = createLayer(layer_input, layers[i], activations[i]);
    net->layers[i] = layer;
    layer_input = layers[i];
  }
  return net;
}

Value *mse(Value *logit, Value *target)
{
  return _pow(_sub(target, logit), 2.0);
}

Value *crossEntropy(Value *logit, Value *target)
{
  /* This is an implementation of a binary cross entropy loss */
  Value *prior = _mul(target, _log(logit));
  Value *post = _mul(_scalarAdd(_scalarMul(target, -1), 1), _log(_scalarAdd(_scalarMul(logit, -1), 1)));
  return _add(prior, post);
}

void sgd(Network *net, double lr, bool bias)
{
  for (size_t n = 0; n < net->num_layers; ++n)
  {
    Layer *layer = net->layers[n];
    for (size_t l = 0; l < layer->num_neurons; ++l)
    {
      Neuron *neuron = layer->neurons[l];

      if (bias == true)
        neuron->bias->data -= lr * neuron->bias->grad;
      // Underflow prevention
      if (neuron->bias->data < EPSILON)
        neuron->bias->data = EPSILON;

      for (size_t i = 0; i < neuron->num_inputs; ++i)
      {
        neuron->weights[i]->data -= lr * neuron->weights[i]->grad;
        // Underflow prevention
        if (neuron->weights[i]->data < EPSILON)
          neuron->weights[i]->data = EPSILON;
      }
    }
  }
}

void zeroGrad(Network *net)
{
  for (size_t n = 0; n < net->num_layers; ++n)
  {
    Layer *layer = net->layers[n];
    for (size_t l = 0; l < layer->num_neurons; ++l)
    {
      Neuron *neuron = layer->neurons[l];
      neuron->bias->grad = 0.0;
      for (size_t i = 0; i < neuron->num_inputs; ++i)
      {
        neuron->weights[i]->grad = 0.0;
      }
    }
  }
}

void freeNet(struct Network *net)
{
  for (size_t i = 0; i < net->num_layers; ++i)
  {
    freeLayer(net->layers[i]);
  }
  free(net);
}
void freeLayer(struct Layer *layer)
{
  for (size_t i = 0; i < layer->num_neurons; ++i)
  {
    freeNeuron(layer->neurons[i]);
  }
  free(layer);
}
void freeNeuron(struct Neuron *neuron)
{
  // freeValue(neuron->bias);
  //  for(size_t i = 0; i < neuron->num_inputs; ++i){
  //      freeValue(neuron->weights[i]);
  //  }
  free(neuron->weights);
  free(neuron);
}
// #endif // NN_IMPLEMENTATION
