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
  rSUB,
  rDIV
} Op;

typedef struct Value
{
  double data;
  double grad;
  size_t num_children;
  struct Value **children;
  Op op;
  int ref_count;
  void (*backward)(struct Value *);
} Value;

Value *initValue(double data);

Value *add(struct Value *v1, struct Value *v2);
Value *scalarAdd(struct Value *v1, double v2);
Value *mul(struct Value *v1, struct Value *v2);
Value *scalarMul(struct Value *v1, double v2);

void noopBackward(struct Value *v);
void addBackwards(struct Value *v);
void mulBackwards(struct Value *v);
void backward(struct Value *v); 

bool valueIn(struct Value *v, struct Value** array);
void buildTopo(struct Value *v, struct Value **topo, struct Value **visisted, size_t *visited_cnt, size_t *topo_cnt);
void appendTopo(struct Value *v, struct Value **topo, size_t *topo_cnt);

void freeValue(struct Value *v);
void printValue(struct Value *v, char *name);

#endif
