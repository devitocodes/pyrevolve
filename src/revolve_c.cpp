#include "../include/revolve_c.h"

extern "C" CRevolve revolve_create_offline(int st, int sn) 
{ 
  CRevolve r;
  r.ptr = new Revolve(st, sn); 
  return r;
}
extern "C" CRevolve revolve_create_multistage(int st, int sn, int sn_ram) 
{ 
  CRevolve r;
  r.ptr = new Revolve(st, sn, sn_ram); 
  return r;
}
extern "C" CRevolve revolve_create_online(int sn) 
{ 
  CRevolve r;
  r.ptr = new Revolve(sn); 
  return r;
}
extern "C" void revolve_destroy(CRevolve r) { delete (Revolve*) r.ptr; }

extern "C" int revolve_adjust(int steps) { return adjust(steps); }
extern "C" int revolve_maxrange(int ss, int tt) { return maxrange(ss, tt); }
extern "C" int revolve_numforw(int steps, int snaps) { return numforw(steps, snaps); }
extern "C" double revolve_expense(int steps, int snaps) { return expense(steps, snaps); }

extern "C" CACTION revolve(CRevolve r) 
{
  int res = ((Revolve*) r.ptr)->revolve(); 
  return (CACTION) res;
}

extern "C" int revolve_getadvances(CRevolve r) { return ((Revolve*) r.ptr)->getadvances(); }
extern "C" int revolve_getcheck(CRevolve r) { return ((Revolve*) r.ptr)->getcheck(); }
extern "C" int revolve_getcheckram(CRevolve r) { return ((Revolve*) r.ptr)->getcheckram(); }
extern "C" int revolve_getcheckrom(CRevolve r) { return ((Revolve*) r.ptr)->getcheckrom(); }
extern "C" int revolve_getcapo(CRevolve r)  { return ((Revolve*) r.ptr)->getcapo(); }
extern "C" int revolve_getfine(CRevolve r)  { return ((Revolve*) r.ptr)->getfine(); }
extern "C" int revolve_getinfo(CRevolve r)  { return ((Revolve*) r.ptr)->getinfo(); }
extern "C" int revolve_getoldcapo(CRevolve r) { return ((Revolve*) r.ptr)->getoldcapo(); }
extern "C" int revolve_getwhere(CRevolve r) 
{ 
  if (((Revolve*) r.ptr)->getwhere())
    return 1;
  else
    return 0;
}
extern "C" void revolve_set_info(CRevolve r, int inf) { ((Revolve*) r.ptr)->set_info(inf); }
extern "C" void revolve_turn(CRevolve r, int final) { ((Revolve*) r.ptr)->turn(final); }

const char* revolve_caction_string(CACTION action) { 
 static const char *CACTION_NAME[] = { "advance", "takeshot", "restore", "firsturn", "youturn", "terminate", "error"};
 return CACTION_NAME[action];
};
