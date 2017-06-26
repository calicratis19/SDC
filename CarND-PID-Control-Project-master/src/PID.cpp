#include "PID.h"

/*
* TODO: Complete the PID class.
*/

PID::PID(double _kp, double _ki, double _kd):
  Kp(_kp),
  Ki(_ki),
  Kd(_kd),
  p_error(0.0),
  i_error(0.0),
  d_error(0.0) 
{}

PID::~PID() {}

void PID::UpdateError(double cte) 
{
  p_error = cte;
  i_error += cte;
  d_error = cte - cte_last;
  cte_last = cte;
}

double PID::TotalError() 
{
  return -Kp*p_error - Kd*d_error - Ki*i_error;
}

