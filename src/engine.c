#include "engine.h"

Value *initValue(double data)
{
  Value *value = calloc(1, sizeof(Value));

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
  return value;
}

Value *_add(struct Value *v1, struct Value *v2)
{
  Value *v = initValue(v1->data + v2->data);
  size_t num_child = 2;
  if (v1 == v2)
    num_child = 1;
  v->children = calloc(num_child, sizeof(Value *));

  if (v->children == NULL)
  {
    fprintf(stderr, "Error: Memory Allocation failed\n");
    return NULL;
  }
  if (v1 == NULL || v2 == NULL)
  {
    fprintf(stderr, "Values couldn't be NULL\n");
    return NULL;
  }

  if (num_child == 1)
  {
    v1->ref_count++;
    v->children[0] = v1;
  }
  else
  {
    v1->ref_count++;
    v2->ref_count++;
    v->children[0] = v1;
    v->children[1] = v2;
  }
  v->num_children = num_child;
  v->op = ADD;
  v->ref_count = 1;
  v->backward = _addBackwards;
  return v;
}
Value *_sub(struct Value *v1, struct Value *v2)
{
  return _add(v1, _scalarMul(v2, -1));
}

Value *_scalarAdd(struct Value *v1, double c)
{
  Value *v2 = initValue(c);
  Value *v = initValue(v1->data + v2->data);
  size_t num_child = 2;
  v->children = calloc(num_child, sizeof(Value *));

  if (v->children == NULL)
  {
    fprintf(stderr, "Error: Memory Allocation failed\n");
    return NULL;
  }

  v1->ref_count++;
  v2->ref_count++;
  v->children[0] = v1;
  v->children[1] = v2;
  v->num_children = num_child;
  v->op = ADD;
  v->ref_count = 1;
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
  size_t num_child = 2;
  if (v1 == v2)
    num_child = 1;
  v->children = calloc(num_child, sizeof(Value *));

  if (v->children == NULL)
  {
    fprintf(stderr, "Error: Memory Allocation failed\n");
    return NULL;
  }
  if (v1 == NULL || v2 == NULL)
  {
    fprintf(stderr, "Values couldn't be NULL\n");
    return NULL;
  }

  if (num_child == 1)
  {
    v1->ref_count++;
    v->children[0] = v1;
  }
  else
  {
    v1->ref_count++;
    v2->ref_count++;
    v->children[0] = v1;
    v->children[1] = v2;
  }
  v->num_children = num_child;
  v->op = MUL;
  v->ref_count = 1;
  v->backward = _mulBackwards;
  return v;
}
Value *_scalarMul(struct Value *v1, double c)
{
  Value *v2 = initValue(c);
  Value *v = initValue(v1->data * v2->data);
  size_t num_child = 2;
  v->children = calloc(num_child, sizeof(Value *));

  if (v->children == NULL)
  {
    fprintf(stderr, "Error: Memory Allocation failed\n");
    return NULL;
  }

  v1->ref_count++;
  v2->ref_count++;
  v->children[0] = v1;
  v->children[1] = v2;
  v->num_children = num_child;
  v->op = MUL;
  v->ref_count = 1;
  v->backward = _mulBackwards;
  return v;
}
Value *_exp(struct Value *v1)
{
  Value *v = initValue(exp(v1->data));
  size_t num_child = 1;
  v->children = calloc(num_child, sizeof(Value *));

  if (v->children == NULL)
  {
    fprintf(stderr, "Error: Memory Allocation failed\n");
    return NULL;
  }

  v1->ref_count++;
  v->children[0] = v1;
  v->num_children = num_child;
  v->op = EXP;
  v->ref_count = 1;
  v->backward = _expBackwards;
  return v;
}
Value *_pow(struct Value *v1, double v2)
{
  Value *v = initValue(pow(v1->data, v2));
  size_t num_child = 1;
  v->children = calloc(num_child, sizeof(Value *));

  if (v->children == NULL)
  {
    fprintf(stderr, "Error: Memory Allocation failed\n");
    return NULL;
  }

  v1->ref_count++;
  v->n = v2;
  v->children[0] = v1;
  v->num_children = num_child;
  v->op = POW;
  v->ref_count = 1;
  v->backward = _powBackwards;
  return v;
}
Value *_log(struct Value *v1){

 Value *v = initValue(log(v1->data));
  size_t num_child = 1;
  v->children = calloc(num_child, sizeof(Value *));

  if (v->children == NULL)
  {
    fprintf(stderr, "Error: Memory Allocation failed\n");
    return NULL;
  }

  v1->ref_count++;
  v->children[0] = v1;
  v->num_children = num_child;
  v->op = LOG;
  v->ref_count = 1;
  v->backward = _powBackwards;
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
  v1->grad += 1/v1->data * v->grad;
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
    ++(*topo_cnt);
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
  Value **topo = calloc(TOPO_SIZE, sizeof(Value *));
  Value **visited = calloc(TOPO_SIZE, sizeof(Value *));
  size_t visited_cnt = 0;
  size_t topo_cnt = 0;
  buildTopo(v, topo, visited, &visited_cnt, &topo_cnt);
  for (size_t i = topo_cnt - 1; i > 0; --i)
  {
    if (topo[i] != NULL)
    {
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
  if (value->ref_count <= 1 && value != NULL)
  {
    free(value);
    value = NULL;
  }
}
void printValue(struct Value *v, char *name)
{
  printf("Value %s.data = %f, .grad = %f\n", name, v->data, v->grad);
}
