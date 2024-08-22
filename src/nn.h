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

Neuron *initNeuron(size_t num_inputs);
Value *forward(Value *inpt);
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

Value *forward(Neuron *n, Value **inpt){
  Value *logit = calloc(1, sizeof(Value));
  for(size_t i = 0; i < n->num_inputs;++i){
    _add(logit, _mul(inpt[i],n->weights[i]));
  }
  _add(logit, n->bias);
  return _tanh(logit);
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
