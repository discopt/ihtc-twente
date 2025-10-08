import sys
import time
import threading
from gurobipy import *

def isFixedVar(var):
  return isinstance(var, int)

def isZeroVar(var):
  return isinstance(var, int) and var == 0

def evalVar(var):
  if isinstance(var, int):
    return float(var)
  else:
    return var.x

class TheaterThread(threading.Thread):

  def __init__(self, algo, adm_sol, days, time_limit_per_day, *arg, **kwargs):
    self._algo = algo
    self._adm_sol = adm_sol
    self._days = days
    self._time_limit_per_day = time_limit_per_day
    super().__init__(*arg, **kwargs)

  def run(self):
    instance = self._algo.instance
    env = Env("", params = {'outputflag': 0, 'threads': 1, 'timelimit': self._time_limit_per_day})
    for d in self._days:
      patients = [ g for g,day in self._adm_sol.guests_day.items() if not instance.guests[g].is_occupant and day == d ]
      assert patients
      theaters = [ t for t,t_data in instance.theaters.items() if t_data.availability[d] > 0 ]
      surgeons = set( instance.patients[p].surgeon for p in patients )

      model = Model("Theaters", env=env)
      model.params.threads = 1
      model.params.nodeLimit = 1 # TODO: replace by callback that ensures that we have at least one solution.

      var_theater_open = {}
      for t in theaters:
        var_theater_open[t] = model.addVar(name=f'TheaterOpen_{t}_on_{d}', vtype=GRB.BINARY)

      var_patient_theater = {}
      for p in patients:
        for t in theaters:
          var_patient_theater[p,t] = model.addVar(name=f'PatTheater_{p}_on_{d}_in_{t}', vtype=GRB.BINARY)

      var_surgeon_theater = {}
      for s in surgeons:
        for t in theaters:
          var_surgeon_theater[s,t] = model.addVar(name=f'SurgeonTheater_{s}_on_{d}_in_{t}', vtype=GRB.BINARY)

      model.update()

      # Each patient goes to one theater.
      for p in patients:
        model.addConstr( quicksum( var_patient_theater[p,t] for t in theaters ) == 1 )

      for t in theaters:
        t_data = instance.theaters[t]
        for p in patients:
          model.addConstr( var_patient_theater[p,t] <= var_theater_open[t] )
        model.addConstr( quicksum( instance.patients[p].surgery_duration * var_patient_theater[p,t] for p in patients ) <= t_data.availability[d] * var_theater_open[t] )

      for t in theaters:
        for p in patients:
          model.addConstr( var_patient_theater[p,t] <= var_surgeon_theater[instance.patients[p].surgeon,t] )

      objective = 0.0
      objective += instance.weights['open_operating_theater'] * quicksum( var_theater_open[t] for t in theaters )
      objective += instance.weights['surgeon_transfer'] * quicksum( var_surgeon_theater[s,t] for s in surgeons for t in theaters )

      model.setObjective(objective)

      model.optimize()

      patients_theater = {}
      if model.solCount:

        for p in patients:
          for t in theaters:
            if var_patient_theater[p,t].x > 0.5:
              self._adm_sol.set_theater(p, t)
      else:
        self._adm_sol.set_theaters_infeasible()


def assignTheaters(instance, algo, adm_sol):

  days = []

  for d in instance.days:
    patients = [ g for g,day in adm_sol.guests_day.items() if not instance.guests[g].is_occupant and day == d ]
    if patients:
      days.append(d)

  num_threads = 2
  print(f'THEA: Solving {len(days)} theater IPs using {num_threads} threads for {adm_sol}.', flush=True)

  adm_sol.pre_theaters()
  time_limit_per_day = 1.0

  threads = []
  for i in range(num_threads):
    thread = TheaterThread(algo, adm_sol, [d for j,d in enumerate(days) if j % num_threads == i ], time_limit_per_day)
    thread.start()
    threads.append(thread)

  for i in range(num_threads):
    threads[i].join()

  if days:
    assert sorted([g for g in adm_sol.guests_day if not instance.guests[g].is_occupant]) == sorted(adm_sol.patients_theater.keys())

  if adm_sol.is_theaters_infeasible:
    print(f'THEA: Infeasible.', flush=True)
  else:
    adm_sol.post_theaters()
    print(f'THEA: Solved theater IPs -> ${adm_sol.costs_open_theaters} + ${adm_sol.costs_surgeon_transfers}.', flush=True)

