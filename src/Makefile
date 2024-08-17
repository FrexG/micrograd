CC := gcc 
CFLAGS := -Wall -Wextra -std=c11 -g

LIBS := `pkg-config --libs --cflags raylib` -lm

TARGET := main 
OBJS := main.o engine.o

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

main.o:main.c engine.h
	$(CC) $(CFLAGS) -o main.o -c main.c $(LIBS)

engine.o:engine.c engine.h
	$(CC) $(CFLAGS) -o engine.o -c engine.c $(LIBS)

clean:
	rm -rf $(TARGET) $(OBJS) 

