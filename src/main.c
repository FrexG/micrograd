#include "engine.h"
#include "viz.h"

int main(void)
{
  InitWindow(W,H, "micrograd");
  SetTargetFPS(30);

  Value *x = initValue(1.0);
  Value *y = initValue(2.0);
  Value *z = initValue(4.0);
  Value *t = add(x, x);
  Value *p = mul(t,y);
  Value *q = mul(z,z);
  Value *s = add(z, p);


  while(!WindowShouldClose()){
    BeginDrawing();
      ClearBackground(BLACK);
      vizGraph(s, W - NODE_W,100);
    EndDrawing();
  }

  CloseWindow();

  s->grad = 1.0;
  backward(s);
  freeValue(s);

  return 0;
}
