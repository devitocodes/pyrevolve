#ifndef _REVOLVE_H_
#define _REVOLVE_H_

enum action { advance, takeshot, restore, firsturn, youturn, terminate, error};

int maxrange(int ss, int tt);
int adjust(int steps);
enum action revolve(int* check,int* capo,int* fine,int snaps,int* info);
int numforw(int steps, int snaps);
double expense(int steps, int snaps);

#endif
