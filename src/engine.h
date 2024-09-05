
#ifndef ENGINE_H
#define ENGINE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define TOPO_SIZE 1000
#define PRINT_V(v) printValue(v, #v)

typedef enum
{
  NONE,
  ADD,
  SUB,
  MUL,
  DIV,
  EXP,
  POW,
  LOG,
  SIGMOID,
  TANH
} OP;

typedef enum
{
  AUTON, // autonomous value, independent and alone
  INPUT,
  OUTPUT,
  OP_OUT, // result of an operation (add, mul, ....)
  LOGIT,  // output of an activation function (tanh, sigmoid, ...)
} VALUE_TYPE;

typedef struct Value
{
  double data;
  double grad;
  double n; // the exponent for power ops
  size_t num_children;
  struct Value **children;
  int ref_count;
  void (*backward)(struct Value *);
  OP op;
  VALUE_TYPE type;
} Value;

Value *initValue(double data);

Value *_add(struct Value *v1, struct Value *v2);
Value *_sub(struct Value *v1, struct Value *v2);
Value *_scalarAdd(struct Value *v1, double v2);
Value *_scalarSub(struct Value *v1, double v2);
Value *_mul(struct Value *v1, struct Value *v2);
Value *_scalarMul(struct Value *v1, double v2);
Value *_pow(struct Value *v1, double v2);
Value *_log(struct Value *v1); // natural log
Value *_div(struct Value *v1, struct Value *v2);
Value *_exp(struct Value *v1);

Value *_sigmoid(struct Value *v);
Value *_tanh(struct Value *v);

void _noopBackward(struct Value *v);
void _addBackwards(struct Value *v);
void _mulBackwards(struct Value *v);
void _expBackwards(struct Value *v);
void _powBackwards(struct Value *v);
void _logBackwards(struct Value *v);
void _sigmoidBackwards(struct Value *v);
void _tanhBackwards(struct Value *v);
void _backward(struct Value *v);

bool valueIn(struct Value *v, struct Value **array);
void buildTopo(struct Value *v, struct Value **topo, struct Value **visisted, size_t *visited_cnt, size_t *topo_cnt);
void appendTopo(struct Value *v, struct Value **topo, size_t *topo_cnt);

void freeValue(struct Value *v);
void printValue(struct Value *v, char *name);

#endif

Value *initValue(double data)
{
  Value *value = malloc(sizeof(Value));

  if (value == NULL)
  {
    fprintf(stderr, "Error: Allocating memory for Value failed\n");
    return NULL;
  }

  value->data = data;
  value->grad = 0.0f;
  value->n = 0.0f;
  value->num_children = 0;
  value->children = NULL;
  value->ref_count = 1;
  value->backward = _noopBackward;
  value->type = AUTON;
  return value;
}

void _initChildren(struct Value *v,struct Value *v1, struct Value *v2){
size_t num_children = 0;
Value **children = NULL;

if((v1 == v2) || v2 == NULL){
  if (v1->type != INPUT){
      num_children = 1;
      children = malloc(num_children* sizeof(Value*));
      v1->ref_count++;
      children[0] = v1;
    }
  }
  else{
      if(v1->type == INPUT && v2->type != INPUT){
        num_children = 1;
        children = malloc(num_children* sizeof(Value*));
        v2->ref_count++;
        children[0] = v2;
      }
      if(v2->type == INPUT && v1->type != INPUT){
        num_children = 1;
        children = malloc(num_children* sizeof(Value*));
        v1->ref_count++;
        children[0] = v1;
      }
      if(v1->type != INPUT && v2->type != INPUT){
        num_children = 2;
        children = malloc(num_children* sizeof(Value*));
        v1->ref_count++;
        v2->ref_count++;
        children[0] = v1;
        children[1] = v2;
      }
  } 
  v->num_children = num_children;
  v->children = children;
}

Value *_add(struct Value *v1, struct Value *v2)
{
  Value *v = initValue(v1->data + v2->data);

  _initChildren(v,v1,v2);

  v->op = ADD;
  v->ref_count = 1;
  v->type = OP_OUT;
  v->backward = _addBackwards;
  return v;
}
Value *_sub(struct Value *v1, struct Value *v2)
{
  /* v1 - v2 */
  return _add(v1, _scalarMul(v2, -1));
}

Value *_scalarAdd(struct Value *v1, double c)
{
  Value *v2 = initValue(c);
  Value *v = initValue(v1->data + v2->data);

  _initChildren(v,v1,v2);
  v->op = ADD;
  v->ref_count = 1;
  v->type = OP_OUT;
  v->backward = _addBackwards;
  return v;
}
Value *_scalarSub(struct Value *v1, double c)
{
  return _scalarAdd(v1, c * -1);
}
Value *_mul(struct Value *v1, struct Value *v2)
{
  Value *v = initValue(v1->data * v2->data);
  
  _initChildren(v,v1,v2);
  v->op = MUL;
  v->ref_count = 1;
  v->type = OP_OUT;
  v->backward = _mulBackwards;
  return v;
}
Value *_scalarMul(struct Value *v1, double c)
{
  Value *v2 = initValue(c);
  Value *v = initValue(v1->data * v2->data);

  _initChildren(v,v1,v2);
  v->op = MUL;
  v->ref_count = 1;
  v->type = OP_OUT;
  v->backward = _mulBackwards;
  return v;
}
Value *_exp(struct Value *v1)
{
  Value *v = initValue(exp(v1->data));
  _initChildren(v,v1,NULL);
  v->op = EXP;
  v->ref_count = 1;
  v->type = OP_OUT;
  v->backward = _expBackwards;
  return v;
}
Value *_pow(struct Value *v1, double v2)
{
  Value *v = initValue(pow(v1->data, v2));
  _initChildren(v,v1,NULL);
  v->n = v2;
  v->op = POW;
  v->ref_count = 1;
  v->type = OP_OUT;
  v->backward = _powBackwards;
  return v;
}
Value *_log(struct Value *v1)
{

  Value *v = initValue(log(v1->data));
  _initChildren(v,v1,NULL);
  v->op = LOG;
  v->ref_count = 1;
  v->type = OP_OUT;
  v->backward = _logBackwards;
  return v;
}
Value *_div(struct Value *v1, struct Value *v2)
{
  // division, can be represented as a multiplication and a power op.
  return _mul(v1, _pow(v2, -1));
}
Value *_sigmoid(struct Value *v1)
{
  return _pow(_scalarAdd(_exp(_scalarMul(v1, -1)), 1), -1);
}

Value *_tanh(struct Value *v1)
{
  
  Value *e2x = _exp(_scalarMul(v1, 2));
  return _div(_scalarSub(e2x, 1), _scalarAdd(e2x, 1));
}

void _noopBackward(struct Value *v) {
  // nothing
};

void _sigmoidBackwards(struct Value *v)
{
  struct Value *v1 = v->children[0];
  v1->grad += v->grad * v->data * (1 - v->data);
}
void _tanhBackwards(struct Value *v)
{
  struct Value *v1 = v->children[0];
  v1->grad += v->grad * (1 - pow(tanh(v->data), 2));
}

void _addBackwards(struct Value *v)
{
  if (v->num_children == 1)
  {
    struct Value *v1 = v->children[0];
    v1->grad += 2 * v->grad;
  }
  else
  {
    struct Value *v1 = v->children[0];
    struct Value *v2 = v->children[1];
    v1->grad += v->grad;
    v2->grad += v->grad;
  }
}

void _mulBackwards(struct Value *v)
{
  if (v->num_children == 1)
  {
    struct Value *v1 = v->children[0];
    v1->grad += 2 * v1->data * v->grad;
  }
  else
  {
    struct Value *v1 = v->children[0];
    struct Value *v2 = v->children[1];
    v1->grad += v2->data * v->grad;
    v2->grad += v1->data * v->grad;
  }
}
void _powBackwards(struct Value *v)
{
  struct Value *v1 = v->children[0];
  v1->grad += v->n * (pow(v1->data, v->n - 1)) * v->grad;
}

void _logBackwards(struct Value *v)
{
  struct Value *v1 = v->children[0];
  v1->grad += 1 / v1->data * v->grad;
}
void _expBackwards(struct Value *v)
{
  struct Value *v1 = v->children[0];
  v1->grad += v->data * v->grad;
}

bool valueIn(struct Value *v, struct Value **list)
{
  for (size_t i = 0; i < TOPO_SIZE; ++i)
  {
    if (v == list[i])
      return true;
  }
  return false;
}

void appendTopo(struct Value *v, struct Value **topo, size_t *topo_cnt)
{
  if (*topo_cnt < TOPO_SIZE)
  {
    topo[*topo_cnt] = v;
    (*topo_cnt) += 1;
  }
}

void buildTopo(struct Value *v, struct Value **topo, struct Value **visited, size_t *visited_cnt, size_t *topo_cnt)
{
  if (v == NULL)
    return;

  if (!valueIn(v, visited))
  {
    if (*visited_cnt < TOPO_SIZE)
    {
      visited[*visited_cnt] = v;
      ++(*visited_cnt);
    }
    for (size_t j = 0; j < v->num_children; ++j)
    {
      if (v->children != NULL)
        buildTopo(v->children[j], topo, visited, visited_cnt, topo_cnt);
    }
    appendTopo(v, topo, topo_cnt);
  }
}

void _backward(struct Value *v)
{
  v->grad = 1.0;
  Value **topo = calloc(TOPO_SIZE, sizeof(Value *));
  Value **visited = calloc(TOPO_SIZE, sizeof(Value *));

  size_t visited_cnt = 0;
  size_t topo_cnt = 0;
  buildTopo(v, topo, visited, &visited_cnt, &topo_cnt);
  for (int i = topo_cnt - 1; i > 0; i--)
  {
    if (topo[i] != NULL)
    {
      if (topo[i]->op != NONE || topo[i]->type != INPUT || topo[i]->type != OUTPUT)
        topo[i]->backward(topo[i]);

    }
  }
  // free buffers
  free(topo);
  free(visited);
}

void freeValue(struct Value *value)
{
  if (value == NULL)
  {
    return;
  }
  value->ref_count--;
  if (value->children != NULL)
  {
    for (size_t i = 0; i < value->num_children; ++i)
    {
      if (value->children[i] != NULL)
        freeValue(value->children[i]);
    }
    free(value->children);
    value->children = NULL;
  }

  if (value->ref_count <= 1 && value->ref_count >= 0 && value != NULL)
  {
    free(value);
    value = NULL;
  }
}
void printValue(struct Value *v, char *name)
{
  printf("Value %s.data = %f, .grad = %f\n", name, v->data, v->grad);
}
