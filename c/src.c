/* -----
*   The function REVOLVE coded below is meant to be used as a        * 
*   "controller" for running a time-dependent applications program   *
*   in the reverse mode with checkpointing described in the paper    *
*   "Achieving logarithmic Growth in temporal and spatial complexity *
*   in reverse automatic differentiation", Optimization Methods and  *
*   Software,  Vol.1 pp. 35-54.                                      *
*   A postscript source of that paper can be found in the ftp sites  *
*        info.mcs.anl.gov and nbtf02.math.tu-dresden.de.             *
*   Apart from REVOLVE this file contains five auxiliary routines    * 
*   NUMFORW, EXPENSE, MAXRANGE, and ADJUST.                          *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   To utilize REVOLVE the user must have procedures for             *
*     - Advancing the state of the modeled system to a certain time. *
*     - Saving the current state onto a stack of snapshots.          *
*     - Restoring the the most recently saved snapshot and           *
*       restarting the forward simulation from there.                *
*     - Initializing the adjoints at the end of forward sweep.       *
*     - Performing one combined forward and adjoint step.            * 
*   Through an encoding of its return value REVOLVE asks the         *
*   calling program to perform one of these 'actions', which we will *
*   refer to as                                                      *
*                                                                    *
*       'advance', 'takeshot', 'restore', 'firsturn' and 'youturn'  .*
*   There are two other return values, namely                        *
*       'terminate'   and     'error'                                *
*   which indicate a regular or faulty termination of the calls      *
*   to REVOLVE.                                                      *
*                                                                    *
*   The action 'firsturn' includes a 'youturn', in that it requires  *
*     -advancing through the last time-step with recording           *
*      of intermediates                                              *
*     -initializing the adjoint values (possibly after               *
*      performing some IO)                                           *
*     -reversing the last time step using the record just written    *
*   The action 'firsturn' is obtained when the difference FINE-CAPO  *
*   has been reduced to 1 for the first time.                        *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   The calling sequence is                                          *
*                                                                    *
*               REVOLVE(CHECK,CAPO,FINE,SNAPS,INFO)                  *
*                                                                    *
*   with the return value being one of the actions to be taken. The  *
*   calling parameters are all integers with the following meaning   *
*                                                                    *
*         CHECK     number of checkpoint being written or retrieved  *
*         CAPO      beginning of subrange currently being processed  *
*         FINE      end of subrange currently being processed        *
*         SNAPS     upper bound on number of checkpoints taken       *
*         INFO      determines how much information will be printed  *
*                   and contains information about an error occured  *
*                                                                    *
*   Since REVOLVE involves only a few integer operations its         *
*   run-time is truly negligible within any nontrivial application.  *
*                                                                    *
*   The parameter SNAPS is selected by the user (possibly with the   *
*   help of the routines EXPENSE and ADJUST described below ) and    *
*   remains unchanged throughout.                                    *
*                                                                    *
*   The pair (CAPO,FINE) always represents the initial and final     *
*   state of the subsequence of time steps currently being traversed *
*   backwards.                                                       *
*                                                                    *
*   The conditions                                                   *
*                    CHECK >= -1      and     CAPO <= FINE           *
*   are necessary and sufficient for a regular response of REVOLVE.  *
*   If either condition is violated the value 'error' is returned.   *
*                                                                    *
*   The first call to REVOLVE must be with CHECK=-1 so that          * 
*   appropriate initializations can be performed internally.         *
*                                                                    *
*   When CHECK =-1 and CAPO = FINE  then 'terminate' is returned as  *
*   action value. This combination necessarily arises after a        *
*   sufficiently large number of calls to REVOLVE, which depends     * 
*   only on the initial difference FINE-CAPO.                        *
*                                                                    *
*   The last parameter INFO determines how much information about    *
*   the actions performed will be printed. When INFO =0 no           *
*   information is sent to standard output. When INFO > 0 REVOLVE    *
*   produces an output that contains a prediction of the number of   *    
*   forward steps and of the factor by which the execution will slow *    
*   down. When an error occurs, the return value of INFO contains    *
*   information about the reason:                                    *
*                                                                    *
*     INFO = 10: number of checkpoints stored exceeds CHECKUP,       *
*                increase constant CHECKUP and recompile             *
*     INFO = 11: number of checkpoints stored exceeds SNAPS, ensure  * 
*                SNAPS greater than 0 and increase initial FINE      *
*     INFO = 12: error occurs in NUMFORW                             *
*     INFO = 13: enhancement of FINE, SNAPS checkpoints stored,      *
*                SNAPS must be increased                             *
*     INFO = 14: number of SNAPS exceeds CHECKUP, increase constant  *
*                CHECKUP and recompile                               *
*     INFO = 15: number of REPS exceeds REPSUP, increase constant    *
*                REPSUP and recompile                                *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   Some further explanations and motivations:                       *
*                                                                    *
*   There is an implicit bound on CHECK through the dimensioning of  *
*   the integer array CH[CHEKUP] with CHECKUP = 64 being the default.*
*   If anybody wants to have that even larger he must change the     *
*   source. Also for the variable REPS an upper bound REPSUP is      *
*   defined. The default value equals 64. If during a call to        *
*   TREEVERSE a (CHECKUP+1)-st checkpoint would normally be called   * 
*   for then control is returned after an appropriate error message. * 
*   When the calculated REPS exceeds REPSUP also an error message    *
*   occurs.                                                          *
*   During the forward sweep the user is free to change the last     *
*   three parameters from call to call, except that FINE may never   *
*   be less than the current value of CAPO. This may be useful when  *
*   the total number of time STEPS to be taken is not a priori       *
*   known. The choice FINE=CAPO+1 initiates the reverse sweep, which * 
*   happens automatically if is left constant as CAPO is eventually  * 
*   moved up to it. Once the first reverse or restore action has     *
*   been taken only the last two parameters should be changed.       *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   The necessary number of forward steps without recording is       *
*   calculated by the function                                       *
*                                                                    *
*                      NUMFORW(STEPS,SNAPS)                          *
*                                                                    *
*   STEPS denotes the total number of time steps, i.e. FINE-CAPO     *
*   during the first call of REVOLVE. When SNAPS is less than 1 an   * 
*   error message will be given and -1 is returned as value.         *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   To choose an appropriated value of SNAPS the function            *
*                                                                    *
*                      EXPENSE(STEPS,SNAPS)                          *
*                                                                    *
*   estimates the run-time factor incurred by REVOLVE for a          *
*   particular value of SNAPS. The ratio NUMFORW(STEPS,SNAPS)/STEPS  *
*   is returned. This ratio corresponds to the run-time factor of    *
*   the execution relative to the run-time of one forward time step. *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   The auxiliary function                                           *
*                                                                    *
*                      MAXRANGE(SNAPS,REPS)                          *
*                                                                    *
*   returns the integer (SNAPS+REPS)!/(SNAPS!REPS!) provided         *
*   SNAPS >=0, REPS >= 0. Otherwise there will be appropriate error  *
*   messages and the value -1 will be returned. If the binomial      *
*   expression is not representable as a  signed 4 byte integer,     *
*   greater than 2^31-1, this maximal value is returned and a        *
*   warning message printed.                                         *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   Furthermore, the function                                        *
*                                                                    *
*                      ADJUST(STEPS)                                 *
*                                                                    *
*   is provided. It can be used to determine a value of SNAPS so     *
*   that the increase in spatial complexity equals approximately the *
*   increase in temporal complexity. For that ADJUST computes a      *
*   return value satisfying SNAPS ~= log_4 (STEPS) because of the    *
*   theory developed in the paper mentioned above.                   *
*                                                                    *
*--------------------------------------------------------------------*/

#include "revolve.h"
#include <stdio.h>
#include <stdlib.h>

#define checkup 64  
#define repsup 64 
#define MAXINT 2147483647

struct 
 {
  int advances;
  int takeshots;
  int commands;
 } numbers;

/* ************************************************************************* */

int numforw(int steps, int snaps)
 {
  int reps, range, num;

  if (snaps < 1) 
   {
    printf(" error occurs in numforw: snaps < 1\n"); 
    return -1;
   }
  if (snaps > checkup)
   {
    printf(" number of snaps=%d exceeds checkup \n",snaps);
    printf(" redefine 'checkup' \n");
    return -1;
   }
  reps = 0;
  range = 1;
  while(range < steps)
   { 
    reps += 1;
    range = range*(reps + snaps)/reps; 
   }
  if (reps > repsup)
   {
    printf(" number of reps=%d exceeds repsup \n",reps);
    printf(" redefine 'repsup' \n");
    return -1;
   }
  num = reps * steps - range*reps/(snaps+1);
  return num;
 }

/* ************************************************************************* */

double expense(int steps, int snaps)
 {
  double ratio;

  if (snaps < 1)
   {
    printf(" error occurs in expense: snaps < 0\n");
    return -1;
   }
  if (steps < 1)
   {
    printf(" error occurs in expense: steps < 0\n");
    return -1;
   }
  ratio = ((double) numforw(steps,snaps));
  if (ratio == -1)
    return -1;
  ratio = ratio/steps;
  return ratio;
 }

/* ************************************************************************* */

int maxrange(int ss, int tt)
 {
  int i, ires;
  double res = 1.0;

  if((tt<0) || (ss<0))
   {
    printf("error in MAXRANGE: negative parameter");
    return -1;
   }
  for(i=1; i<= tt; i++)
   {  
    res *= (ss + i);
    res /= i;
    if (res > MAXINT)
     { 
      ires=MAXINT;
      printf("warning from MAXRANGE: returned maximal integer %d\n",ires);
      return ires;
     }
   }
  ires = res;
  return ires;
 }

/* ************************************************************************* */

int adjust(int steps)
 {
  int snaps, s, reps;

  snaps = 1;
  reps = 1;
  s = 0;
  while( maxrange(snaps+s, reps+s) > steps ) 
    s--;
  while( maxrange(snaps+s, reps+s) < steps ) 
    s++;
  snaps += s;
  reps += s ;
  s = -1;
  while( maxrange(snaps,reps) >= steps )
   {
    if (snaps > reps)
     { 
      snaps -= 1; 
      s = 0; 
     }
    else 
     { 
      reps -= 1; 
      s = 1; 
     }
   }
  if ( s == 0 ) 
    snaps += 1 ;
  if ( s == 1 ) 
    reps += 1;
  return snaps;
 }

/* ************************************************************************* */ 

enum action revolve(int* check,int* capo,int* fine,int snaps,int* info)
 {
  static int  turn, reps, range, ch[checkup], oldsnaps, oldfine;
  int ds, oldcapo, num, bino1, bino2, bino3, bino4, bino5;
    /* (*capo,*fine) is the time range currently under consideration */
    /* ch[j] is the number of the state that is stored in checkpoint j */

  numbers.commands += 1;
  if ((*check < -1) || (*capo > *fine)) 
    return error; 
  if ((*check == -1) && (*capo < *fine))
   {
    if (*check == -1) 
      turn = 0;   /* initialization of turn counter */
    *ch = *capo-1;   
   }
  switch(*fine-*capo)
   { 
    case 0:   /* reduce capo to previous checkpoint, unless done  */
      if(*check == -1 || *capo==*ch )
       { 
        *check -= 1;
        if (*info > 0)  
         { 
          printf(" \n advances: %5d",numbers.advances);
          printf(" \n takeshots: %4d",numbers.takeshots);
          printf(" \n commands: %5d \n",numbers.commands);
         } 
        return terminate;
       }
      else
       { 
        *capo = ch[*check];
        oldfine = *fine;
        return restore;
       } 
    case 1:  /* (possibly first) combined forward/reverse step */ 
      *fine -= 1;
      if(*check >= 0 && ch[*check] == *capo) 
        *check -= 1; 
      if(turn == 0)
       {
        turn = 1;
        oldfine = *fine;
        return firsturn; 
       }
      else 
       { 
        oldfine = *fine;
        return youturn; 
       }
    default:         
      if(*check == -1 || ch[*check] != *capo) 
       { 
        *check += 1 ; 
        if(*check >= checkup)
	 { 
	  *info = 10;
          return error;
	 }
        if(*check+1 > snaps)
         {
          *info = 11;
          return error;
         }
        ch[*check] = *capo;
        if (*check == 0) 
         {
          numbers.advances = 0;
          numbers.takeshots = 0;
          numbers.commands = 1;
          oldsnaps = snaps;
          if (snaps > checkup)
           {
            *info = 14;
            return error;
           }
          if (*info > 0) 
           {
            num = numforw(*fine-*capo,snaps);
            if (num == -1) 
             {
              *info = 12;
              return error;
             }
            printf(" prediction of needed forward steps: %8d => \n",num);
            printf(" slowdown factor: %8.4f \n\n",((double) num)/(*fine-*capo));
           }
         }
        numbers.takeshots += 1;
        oldfine = *fine;
        return takeshot; 
       }
      else
       { 
        if ((oldfine < *fine) && (snaps == *check+1))
         { 
          *info = 13;
          return error;
         } 
        oldcapo = *capo;
        ds = snaps - *check;
        if (ds < 1)
         {
          *info = 11;
          return error;
         }
        reps = 0;
        range = 1;
        while(range < *fine - *capo) 
         { 
          reps += 1;
          range = range*(reps + ds)/reps; 
         }
        if (reps > repsup) 
         { 
          *info = 15;
          return error;
         }
        if (snaps != oldsnaps)
         { 
           if (snaps > checkup)
            {
             *info = 14;
             return error;
            }
         }
        bino1 = range*reps/(ds+reps);
        bino2 = (ds > 1) ? bino1*ds/(ds+reps-1) : 1;
        if (ds == 1)
          bino3 = 0;
        else
          bino3 = (ds > 2) ? bino2*(ds-1)/(ds+reps-2) : 1;
        bino4 = bino2*(reps-1)/ds;
        if (ds < 3)
          bino5 = 0;
        else
          bino5 = (ds > 3) ? bino3*(ds-2)/reps : 1;
     
        if (*fine-*capo <= bino1 + bino3)
          *capo = *capo+bino4;
        else 
         {
          if (*fine-*capo >= range - bino5) 
            *capo = *capo + bino1; 
          else 
             *capo = *fine-bino2-bino3;
         }
        if (*capo == oldcapo) 
          *capo = oldcapo+1;  
        numbers.advances = numbers.advances + *capo - oldcapo; 
        oldfine = *fine;
        return advance;
       }          
    }
   free(ch);
 }     

