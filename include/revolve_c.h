#ifndef ADJ_REVOLVE_H
#define ADJ_REVOLVE_H

#ifdef __cplusplus
#include <iostream>
#include "revolve.h"
#endif

typedef struct 
{
  void *ptr;
} CRevolve;

typedef enum 
{ 
  CACTION_ADVANCE, CACTION_TAKESHOT, CACTION_RESTORE, CACTION_FIRSTRUN, CACTION_YOUTURN, CACTION_TERMINATE, CACTION_ERROR
} CACTION;

#ifdef __cplusplus
extern "C" { 
#endif
CRevolve revolve_create_offline(int st, int sn);
CRevolve revolve_create_multistage(int st, int sn, int sn_ram);
CRevolve revolve_create_online(int sn);
void revolve_destroy(CRevolve r);
CACTION revolve(CRevolve r); 
int revolve_adjust(int steps);
int revolve_maxrange(int ss, int tt);
int revolve_numforw(int steps, int snaps);
double revolve_expense(int steps, int snaps);
int revolve_getadvances(CRevolve r);
int revolve_getcheck(CRevolve r);
int revolve_getcheckram(CRevolve r);
int revolve_getcheckrom(CRevolve r);
int revolve_getcapo(CRevolve r);
int revolve_getfine(CRevolve r);
int revolve_getinfo(CRevolve r);
int revolve_getoldcapo(CRevolve r);
int revolve_getwhere(CRevolve r);
void revolve_setinfo(CRevolve r, int inf);
void revolve_turn(CRevolve r, int final);
const char* revolve_caction_string(CACTION action);
#ifdef __cplusplus
}
#endif

#endif
