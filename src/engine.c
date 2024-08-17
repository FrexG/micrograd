#include "engine.h"



Value *initValue(double data)
{
  Value *value = calloc(1, sizeof(Value));

  if (value == NULL)
  {
    fprintf(stderr, "Error: Memory Allocation failed\n");
    return NULL;
  }
  value->data = data;
  value->grad = 0.0f;
  value->num_children = 0;
  value->children = NULL;
  value->ref_count = 1;
  value->backward = noopBackward;
  return value;
}


Value *add(struct Value *v1, struct Value *v2)
{
  Value *v = initValue(v1->data + v2->data);
  size_t num_child = 2;
  if(v1 == v2)
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
  v->backward = addBackwards;
  return v;
}

Value *mul(struct Value *v1, struct Value *v2)
{
  Value *v = initValue(v1->data * v2->data);
  size_t num_child = 2;
  if(v1 == v2)
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
  v->backward = mulBackwards;
  return v;
}


void noopBackward(struct Value *v)
{
  (void*)v;
};

void addBackwards(struct Value *v)
{
  if(v->num_children == 1){
    struct Value *v1 = v->children[0];
    v1->grad += 2*v->grad;
  }
  else{
    struct Value *v1 = v->children[0];
    struct Value *v2 = v->children[1];
    v1->grad += v->grad;
    v2->grad += v->grad;
  }
}

void mulBackwards(struct Value *v)
{
  if(v->num_children == 1){
    struct Value *v1 = v->children[0];
    v1->grad += 2*v1->data * v->grad;
  }
  else{
    struct Value *v1 = v->children[0];
    struct Value *v2 = v->children[1];
    v1->grad += v2->data * v->grad;
    v2->grad += v1->data * v->grad;
  }
}


bool valueIn(struct Value* v, struct Value** list){
  for(size_t i = 0; i < TOPO_SIZE; ++i){
    if(v == list[i])
      return true;
  }
  return false;
}

void appendTopo(struct Value *v, struct Value **topo, size_t *topo_cnt){
  if(*topo_cnt < TOPO_SIZE){
    topo[*topo_cnt] = v;
    ++(*topo_cnt);
  }
}

void buildTopo(struct Value *v, struct Value **topo, struct Value **visited, size_t *visited_cnt, size_t *topo_cnt){
  if(v == NULL)
    return;
  if(!valueIn(v, visited)){
    if(*visited_cnt < TOPO_SIZE){
      visited[*visited_cnt] = v;
      ++(*visited_cnt);
    }
    for(size_t j = 0; j < v->num_children; ++j){
      if(v->children != NULL)
        buildTopo(v->children[j], topo, visited, visited_cnt, topo_cnt);
    }
    appendTopo(v,topo, topo_cnt);
  }

}
void backward(struct Value *v){
  Value **topo = calloc(TOPO_SIZE, sizeof(Value*));
  Value **visited = calloc(TOPO_SIZE, sizeof(Value*));
  size_t visited_cnt = 0;
  size_t topo_cnt = 0;
  buildTopo(v, topo, visited, &visited_cnt, &topo_cnt);
  for(size_t i = topo_cnt -1; i > 0; --i){
    if(topo[i] != NULL){
      PRINT_V(topo[i]);
      topo[i]->backward(topo[i]);
    }
  }
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
    printf("Has Children\n");
    printf("\t%f\n", value->data);
    printf("\tref count = %d\n", value->ref_count);

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
    printf("free %f\n", value->data ? value->data : 0.0);
    free(value);
    value = NULL;
  }
}
void printValue(struct Value *v, char *name)
{
  printf("Value %s.data = %f, .grad = %f\n", name, v->data, v->grad);
}
