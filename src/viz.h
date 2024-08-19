#ifndef VIZ_H
#define VIZ_H

#include <raylib.h> // for grapth visualization
#include <stdlib.h>

#define W 1000
#define H 500

#define NODE_W 50
#define NODE_H 20

typedef struct Node{
  int x;
  int y;
  double data;
  size_t num_children;
  struct Node **children;
}Node;

void drawNodes(Value *v, Node *n,int x, int y){
  if (v == NULL)
    return;

  n->x = x;
  n->y = y;
  n->data = v->data;
  if (v->children != NULL)
  {
    n->num_children = v->num_children;
    n->children = calloc(n->num_children,sizeof(Node*));
    int pos = 1;
    for (size_t i = 0; i < v->num_children; ++i)
    {
      if(v->children[i] != NULL){
        n->children[i] = calloc(1,sizeof(Node));
        pos *= -1; 
        drawNodes(v->children[i],n->children[i],x - NODE_W * 2, y + NODE_H * i * pos * 2);
      }
    }
  }
  DrawRectangleLines(x, y, NODE_W, NODE_H, RED);
  char value_text[32];
  snprintf(value_text, sizeof(value_text), "%.2f", v->data);
  DrawText(value_text, x + 5, y + 5, 10, WHITE);
  switch (v->op)
  {
  case ADD:
    DrawText("ADD", x + 5, y + 20, 10, WHITE);
    break;
  case MUL:
    DrawText("MUL", x + 5, y + 20, 10, WHITE);
    break;
  case EXP:
    DrawText("EXP", x + 5, y + 20, 10, WHITE);
    break;
  default:
    DrawText("NONE", x + 5, y + 20, 10, WHITE);
    break;
  }
}
void drawEdges(Node *n){
  if(n == NULL)
    return;
  if(n->children != NULL){
    for(size_t i = 0; i < n->num_children; ++i){
      drawEdges(n->children[i]);
      Node *child = n->children[i];
      //DrawLine(n->x,n->y, child->x, child->y,BLUE);
      Vector2 start_pos = {n->x,n->y + NODE_H/2};
      Vector2 end_pos = {child->x + NODE_W, child->y + NODE_H/2};
      DrawLineBezier(start_pos, end_pos, 4.0f, BLUE);
    }
  }
  return;
}
void vizGraph(Value *v, int x, int y)
{
  Node *n = calloc(1, sizeof(Node));
  drawNodes(v, n,x, y);
  drawEdges(n);
}
#endif