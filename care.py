import sys
import time
import threading
from gurobipy import *

class CareCostsTask:

  def __init__(self, instance, patient, day, result):
    self._instance = instance
    self._patient = patient
    self._day = day
    self._result = result

  def run(self):
#    print(f'Computing lower bound on care costs if {self._patient} is admitted on {self._day}.', flush=True)

    p_data = self._instance.patients[self._patient]
    first_shift = self._instance.day_to_shift(self._day)
    beyond_shift = min(self._instance.numShifts, first_shift + self._instance.numShiftTypes * p_data.length_of_stay)
    skill_level_required = p_data.skill_level_required

    env = Env("", params = {
      'outputflag': 0,
      'threads': 1,
      })
    model = Model("CareCostsTask", env=env)

    var_nurse_patient = {}
    for n,n_data in self._instance.nurses.items():
      var_nurse_patient[n] = model.addVar(vtype=GRB.BINARY, obj=self._instance.weights['continuity_of_care'])

    var_nurse_shift = {}
    for n,n_data in self._instance.nurses.items():
      for s in n_data.shifts:
        if s >= first_shift and s < beyond_shift:
          var_nurse_shift[n,s] = model.addVar(vtype=GRB.BINARY, obj=self._instance.weights['room_nurse_skill'] * max(0, skill_level_required[s - first_shift] - n_data.skill_level))

    model.update()

    # Each patient is treated by a nurse in each shift.
    for s in range(first_shift, beyond_shift):
      model.addConstr( quicksum( var_nurse_shift.get((n, s), 0.0) for n in self._instance.nurses ) >= 1 )

    # If a nurse treats a patient then it must be enabled.
    for key,var in var_nurse_shift.items():
      n = key[0]
      model.addConstr( var <= var_nurse_patient[n], f'Imply_{n}_{key[1]}' )

    model.optimize()
 
    self._result[self._patient,self._day] = int(model.objVal + 0.5)

#    sys.stderr.write(f'Lower bound on care costs if {self._patient} is admitted on {self._day} is {model.objVal}.\n')
#    sys.stderr.flush()


