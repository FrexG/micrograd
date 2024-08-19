#include "engine.h"
#include "viz.h"

int main(void)
{
  // Set the window to be resizable
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(W,H, "micrograd");
  SetTargetFPS(30);

  Value *x = initValue(1.0);
  Value *y = initValue(2.0);
  Value *z = initValue(4.0);

  Value *s = _scalarMul(_add(_add(z,_mul(_add(z, x),y)),_exp(_mul(y,z))),10);


  while(!WindowShouldClose()){
    int screenWidth = GetScreenWidth();
    int screenHeight = GetScreenWidth();

    BeginDrawing();
      ClearBackground(BLACK);
      vizGraph(s, screenWidth - NODE_W,100);
    EndDrawing();
  }

  CloseWindow();

  s->grad = 1.0;
  backward(s);
  freeValue(s);

  return 0;
}
