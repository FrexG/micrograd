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

// Construction
Neuron *initNeuron(size_t num_inputs);
Layer *creatLayer(size_t num_inputs,size_t num_neurons);
// Computation
Value **forward(Network*net, Value **inpt);
// Cost
Value *mse(Value **logits, Value **targets, size_t num_outputs);
// Optimizers
void sgd(Network *net, double lr);
void zeroGrad(Network *net);

void freeNeuron(struct Neuron *neuron);
void freeNet(struct Network *net);
void freeLayer(struct Layer *layer);

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
  Value **activations = inpt;

  for(size_t n = 0; n < net->num_layers; ++n){
    Layer *layer = net->layers[n];
    logits = (Value**) calloc(layer->num_neurons, sizeof(Value));

    for(size_t l = 0; l < layer->num_neurons; ++l){
      logits[l] = initValue(0.0);
      Neuron *neuron = layer->neurons[l];
      for(size_t i = 0; i < neuron->num_inputs; ++i){
          logits[l] = _add(logits[l],_mul(activations[i],neuron->weights[i]));
      }
      logits[l] = _add(logits[l],neuron->bias);
      logits[l] = _tanh(logits[l]);
    }
    activations = logits;
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
  net->layers = (Layer**) calloc(num_layers,sizeof(Layer*));

  // Create layers
  size_t layer_input = num_inputs;
  for(size_t i = 0; i < num_layers; ++i){
    Layer *layer = createLayer(layer_input,layers[i]);
    net->layers[i] = layer;
    layer_input = layers[i];
  }
  return net;
}

Value *mse(Value **logits, Value **targets, size_t num_outputs){
  Value *loss = initValue(0.0);
  for(size_t i = 0; i < num_outputs; ++i){
    loss = _pow(_sub(targets[i], logits[i]),2.0);
  }
  return loss;
}

void sgd(Network *net, double lr){
  for(size_t n = 0; n < net->num_layers; ++n){
    Layer *layer = net->layers[n];
    for(size_t l = 0; l < layer->num_neurons; ++l){
      Neuron *neuron = layer->neurons[l];
      neuron->bias->data -= lr * neuron->bias->grad;
      for(size_t i = 0; i < neuron->num_inputs; ++i){
        neuron->weights[i]->data -= lr * neuron->weights[i]->grad;
      }
    }
  }
}

void zeroGrad(Network *net){
  for(size_t n = 0; n < net->num_layers; ++n){
    Layer *layer = net->layers[n];
    for(size_t l = 0; l < layer->num_neurons; ++l){
      Neuron *neuron = layer->neurons[l];
      neuron->bias->grad = 0.0;
      for(size_t i = 0; i < neuron->num_inputs; ++i){
        neuron->weights[i]->grad = 0.0;
      }
    }
  }
}

void freeNet(struct Network *net){
  for(size_t i = 0; i < net->num_layers; ++i){
    freeLayer(net->layers[i]);
  }
  free(net);
}

void freeLayer(struct Layer *layer){
  for(size_t i = 0; i < layer->num_neurons; ++i){
    freeNeuron(layer->neurons[i]);
  }
  free(layer);
}
void freeNeuron(struct Neuron *neuron){
  free(neuron->weights);
  free(neuron);
}
//#endif // NN_IMPLEMENTATION
