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
} Op;

typedef struct Value
{
  double data;
  double grad;
  double n; // the exponent for power ops
  size_t num_children;
  struct Value **children;
  Op op;
  int ref_count;
  void (*backward)(struct Value *);
} Value;

Value *initValue(double data);

Value *_add(struct Value *v1, struct Value *v2);
Value *_sub(struct Value *v1, struct Value *v2);
Value *_scalarAdd(struct Value *v1, double v2);
Value *_scalarSub(struct Value *v1, double v2);
Value *_mul(struct Value *v1, struct Value *v2);
Value *_scalarMul(struct Value *v1, double v2);
Value *_pow(struct Value *v1, double v2);
Value *_div(struct Value *v1, struct Value *v2);
Value *_exp(struct Value *v1);

Value *_sigmoid(struct Value *v);
Value *_tanh(struct Value *v);

void noopBackward(struct Value *v);
void addBackwards(struct Value *v);
void mulBackwards(struct Value *v);
void expBackwards(struct Value *v);
void powBackwards(struct Value *v);
void backward(struct Value *v);

bool valueIn(struct Value *v, struct Value **array);
void buildTopo(struct Value *v, struct Value **topo, struct Value **visisted, size_t *visited_cnt, size_t *topo_cnt);
void appendTopo(struct Value *v, struct Value **topo, size_t *topo_cnt);

void freeValue(struct Value *v);
void printValue(struct Value *v, char *name);

#endif
