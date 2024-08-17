#include "engine.h"
#include <raylib.h> // for grapth visualization

#define W 600
#define H 200

void vizGraph(Value *v, int x, int y)
{
  if (v == NULL)
    return;

  if (v->children != NULL)
  {
    for (int i = 0; i < 2; ++i)
    {
      vizGraph(v->children[i], x - 50, y - 50 * i);
    }
  }
  DrawRectangleLines(x, y, 10, 10, RED);
}

int main(void)
{
  // InitWindow(W,H, "micrograd");
  // SetTargetFPS(30);

  Value *x = initValue(1.0);
  Value *y = initValue(2.0);
  Value *z = initValue(4.0);
  Value *t = add(x, x);
  Value *p = mul(t,y);
  Value *q = mul(z,z);
  Value *s = add(z, p);


  // while(!WindowShouldClose()){
  //   BeginDrawing();
  //     ClearBackground(RAYWHITE);
  //     //vizGraph(p, 400,100);
  //   EndDrawing();
  // }

  // CloseWindow();
  PRINT_V(y);
  PRINT_V(x);
  PRINT_V(z);
  PRINT_V(t);
  PRINT_V(p);
  PRINT_V(s);

  s->grad = 1.0;

  //s->backward(s);
  //p->backward(p);
  //t->backward(t);
  //z->backward(z);
  //y->backward(y);
  //x->backward(x);
  backward(s);

  PRINT_V(y);
  PRINT_V(x);
  PRINT_V(z);
  PRINT_V(t);
  PRINT_V(p);
  PRINT_V(s);
  freeValue(s);

  return 0;
}
