#ifndef NN_H
#define NN_H
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "engine.h"

typedef struct Neuron{
  size_t num_inputs;
  struct Value **weights;
  struct Value *bias;
}Neuron;

typedef struct Layer{
  size_t num_neurons;
  struct Neuron **neurons;
} Layer;

typedef struct Network{
  size_t num_inputs;
  size_t num_outputs;
  size_t num_layers;
  struct Layer **layers;
} Network;

Neuron *initNeuron(size_t num_inputs);
Value **forward(Value *inpt);
Layer *creatLayer(size_t num_inputs,size_t num_neurons);
void freeNeuron(struct Neuron *neuron);

#endif

//#ifdef NN_IMPLEMENTATION

Neuron *initNeuron(size_t num_inputs){
  Neuron *neuron = (Neuron*) calloc(1, sizeof(Neuron));

  if(neuron == NULL){
    fprintf(stderr, "Error allocating memory for a neuron\n");
    return NULL;
  }
  neuron->num_inputs = num_inputs;
  neuron->weights = (Value**) calloc(num_inputs, sizeof(Neuron*));

  for(size_t i = 0; i < num_inputs; ++i){
    double data = (double)rand() / RAND_MAX;
    Value *w = initValue(data);
    neuron->weights[i] = w;
  }
  
  double data = (double)rand() / RAND_MAX;
  neuron->bias = initValue(data);

  return neuron;
}

Value **forward(Network*net, Value **inpt){
  Value **logits;

  for(size_t n = 0; n < net->num_layers; ++n){
    Layer *layer = net->layers[n];
    logits = (Value**) calloc(layer->num_neurons, sizeof(Value));

    for(size_t l = 0; l < layer->num_neurons; ++l){
      Neuron *neuron = layer->neurons[l];
      for(size_t i = 0; i < neuron->num_inputs; ++i){
        _add(logits[l],_mul(inpt[i],neuron->weights[i]));
      }
      _add(logits[l],neuron->bias);
      _tanh(logits[l]);
    }
  }
  return logits;
}

Layer *createLayer(size_t num_inputs,size_t num_neurons){
  // creat a layer
  Layer *layer = (Layer*) calloc(1, sizeof(Layer));
  if(layer == NULL){
    fprintf(stderr, "Error allocating memory for a layer\n");
    return NULL;
  }
  layer->num_neurons = num_neurons;
  layer->neurons = (Neuron**) calloc(num_neurons, sizeof(Neuron*));

  for(size_t i = 0; i < num_neurons; ++i){
    Neuron *n = initNeuron(num_inputs);
    layer->neurons[i] = n;
  }
  return layer;
}

Network *createNetwork(size_t num_inputs, size_t num_outputs, size_t num_layers, size_t *layers){
  Network *net = (Network*) calloc(1, sizeof(Network));
  if(net == NULL){
    fprintf(stderr, "Error allocating memory for a network\n");
    return NULL;
  }
  // initialize
  net->num_inputs = num_inputs;
  net->num_outputs = num_outputs;
  net->num_layers = num_layers;

  // Create layers
  size_t layer_input = num_inputs;
  for(size_t i = 0; i < num_layers; ++i){
    Layer *layer = createLayer(layer_input,layers[i]);
    layer_input = layers[i];
  }
  return net;
}

void freeNeuron(struct Neuron *neuron){
  // free the bias value
  freeValue(neuron->bias);
  for(size_t i = 0; i < neuron->num_inputs; ++i){
    freeValue(neuron->weights[i]);
  }
  free(neuron->weights);
  free(neuron);
}
//#endif // NN_IMPLEMENTATION
