import sys
import threading
import time
from gurobipy import *
from data import *
from solution import *

def isFixedVar(var):
  return isinstance(var, int)

def isZeroVar(var):
  return isinstance(var, int) and var == 0

def evalVar(var):
  if isinstance(var, int):
    return float(var)
  else:
    return var.x

class AdmissionsThread(threading.Thread):

  def __init__(self, instance, patients_days_care_bound, room_capacity_reduction, label, algo, time_limit, initial_solution=None, *arg, **kwargs):
    self._instance = instance
    self._patients_days_care_bound = copy.copy(patients_days_care_bound)
    self._interrupted = False
    self._room_capacity_reduction = room_capacity_reduction
    self._label = label
    self._algo = algo
    self._time_limit = time_limit
    self._initial_solution = initial_solution
    super().__init__(*arg, **kwargs)

  @property
  def room_capacity_reduction(self):
    return self._room_capacity_reduction

  def interrupt(self):
    self._interrupted = True

  def run(self):
    if self._time_limit < 1.0:
      print(f'ADM{self._label}: Skipping due to time limit.', flush=True)
      return


    instance = self._instance
    guests_admissionDays = { g: set() for g in instance.guests } | { o: { 0 } for o in instance.occupants }
    for p,d in self._patients_days_care_bound:
      guests_admissionDays[p].add(d)
    total_nurse_capacities = instance.getTotalNurseCapacities()

    print(f'ADM{self._label}: Starting to solve MIP with a time limit of {self._time_limit:.1f}s.', flush=True)
    
    env = Env("", params = {'outputflag': 0, 'threads': 1, 'timelimit': self._time_limit})
    model = Model("Admissions", env=env)
    model._instance = instance
    model._thread = self

    model._var_patient_day = {}
    for p,p_data in instance.patients.items():
      for d in instance.days:
        if (p,d) not in self._patients_days_care_bound:
          model._var_patient_day[p,d] = 0
        else:
          model._var_patient_day[p,d] = model.addVar(name=f'PatDay_{p}_day{d}', vtype=GRB.BINARY)
#    sys.stderr.write(f'ADM{self._label}: #patient-day variables: {sum(1 for key,value in model._var_patient_day.items() if not isFixedVar(value))}\n')

    # Fake variables for occupants.
    model._var_occupant_day = { (o,d): 0 for o in instance.occupants for d in instance.days }
    for o,o_data in instance.occupants.items():
      model._var_occupant_day[o,0] = 1

    model._var_guest_day = { **model._var_patient_day, **model._var_occupant_day }

    # Variables for staying
    model._var_guest_day_staying = {}
    for g,g_data in instance.guests.items():
      for d in instance.days:
        fixed0 = True
        fixed1 = False
        for d_adm in guests_admissionDays[g]:
          if d >= d_adm and d <= d_adm + g_data.length_of_stay - 1:
            if isZeroVar(model._var_guest_day[g,d_adm]):
              pass
            if isFixedVar(model._var_guest_day[g,d_adm]):
              assert not fixed1
              if fixed0:
                fixed0,fixed1 = False,True
            else:
              fixed0,fixed1 = False,False
              break
        if fixed0:
          model._var_guest_day_staying[g,d] = 0
        elif fixed1 and g_data.is_mandatory:
          model._var_guest_day_staying[g,d] = 1
        else:
          model._var_guest_day_staying[g,d] = model.addVar(name=f'GuestDayStaying_{g}_day{d}', vtype=GRB.BINARY)
#    sys.stderr.write(f'ADM{self._label}: #guest-day-staying variables: {sum(1 for key,value in model._var_guest_day_staying.items() if not isFixedVar(value))}\n')

    # Patient unscheduled?
    model._var_patient_unscheduled = {}
    for p,p_data in instance.patients.items():
      if p_data.is_mandatory:
        model._var_patient_unscheduled[p] = 0
      else:
        model._var_patient_unscheduled[p] = model.addVar(name=f'PatUnsched_{p}', vtype=GRB.BINARY)
#    sys.stderr.write(f'ADM{self._label}: #patient-unscheduled variables: {sum(1 for key,value in model._var_patient_unscheduled.items() if not isFixedVar(value))}\n')

    # Nurse workload excess.
    model._var_shift_workload_excess = {}
    for s in instance.shifts:
      model._var_shift_workload_excess[s] = model.addVar(name=f'WorkloadExcess_{instance.shift_label(s)}')
#    sys.stderr.write(f'ADM{self._label}: #shift-workload-excess variables: {sum(1 for key,value in model._var_patient_unscheduled.items() if not isFixedVar(value))}\n')

    # Theater openings.
    model._var_theater_opening = {}
    for d in instance.days:
      theater_capacities = { }
      for ot,ot_data in instance.theaters.items():
        cap = ot_data.availability[d]
        if cap > 0:
          theater_capacities[cap] = theater_capacities.get(cap, 0) + 1
      for cap in theater_capacities.keys():
        model._var_theater_opening[d,cap] = model.addVar(name=f'TheaterOpening_{d}_{cap}', ub=theater_capacities[cap], vtype=GRB.INTEGER)
#    sys.stderr.write(f'ADM{self._label}: #theater-opening variables: {len(model._var_theater_opening)}\n')

    model.update()

#    sys.stderr.write(f'ADM{self._label}: #variables: {model.numVars}\n')

    # Link model._var_patient_day and model._var_guest_day_staying
    cons_guest_day_staying = {}
    for p,p_data in instance.patients.items():
      for d in instance.days:
        if isFixedVar(model._var_guest_day_staying[p,d]):
          continue
        cons_guest_day_staying[p,d] = model.addConstr(
          model._var_guest_day_staying[p,d] ==
          quicksum( model._var_patient_day[p,dp] for dp in instance.days if d >= dp and d <= dp + p_data.length_of_stay - 1 ),
          name=f'GuestDayStaying_{p}_day{d}')
#    sys.stderr.write(f'ADM{self._label}: #guest-day-staying constraints: {len(cons_guest_day_staying)}\n')

    # H3:  Surgeon overtime: The maximum daily surgery time of a surgeon must not be exceeded.
    cons_surgeon_overtime = {}
    for s,s_data in instance.surgeons.items():
      for d in instance.days:
        cons_surgeon_overtime[s,d] = model.addConstr(
          quicksum( p_data.surgery_duration * model._var_patient_day[p,d] for p,p_data in instance.patients.items() if p_data.surgeon == s )
          <= s_data.max_surgery_time[d],
          name=f'SurgeonOvertime_{s}_day{d}')
#    sys.stderr.write(f'ADM{self._label}: #surgeon-overtime constraints: {len(cons_surgeon_overtime)}\n')

    # H5: Mandatory versus optional patients: All mandatory patients must be admitted within the scheduling period, whereas optional patients may be postponed to future scheduling periods.
    cons_patient_scheduled = {}
    for p,p_data in instance.patients.items():
      cons_patient_scheduled[p] = model.addConstr(
        quicksum( model._var_patient_day[p,d] for d in instance.days ) + model._var_patient_unscheduled[p] == 1,
        name=f'PatientScheduled_{p}')
#    sys.stderr.write(f'ADM{self._label}: #patient-scheduled constraints: {len(cons_patient_scheduled)}\n')

    # Aggregated H7: total room capacity: The number of patients in all rooms in each day cannot exceed the total room capacity.
    cons_room_capacity = {}
    total_capacity = -self._room_capacity_reduction
    for r,r_data in instance.rooms.items():
      total_capacity += r_data.capacity
    for d in instance.days:
      cons_room_capacity[d] = model.addConstr(
        quicksum( model._var_guest_day_staying[g,d] for g in instance.guests )
        <= total_capacity, name=f'RoomCapacity_day{d}')
#    sys.stderr.write(f'ADM{self._label}: #room-capacity constraints: {len(cons_room_capacity)}\n')

    # Total nurse workload excess
    cons_total_nurse_workload = {}
    for s in instance.shifts:
      cons_total_nurse_workload[s] = model.addConstr(
        model._var_shift_workload_excess[s] + total_nurse_capacities[s]
          >= quicksum( p_data.workload_produced[s - 3*d] * model._var_patient_day[p,d] for d in instance.days for p,p_data in instance.patients.items() if s >= 3*d and s < 3*(d + p_data.length_of_stay) )
          + sum( o_data.workload_produced[s] for o,o_data in instance.occupants.items() if s < 3*o_data.length_of_stay ),
          name=f'TotalNurseWorkload_{s}')
#    sys.stderr.write(f'ADM{self._label}: #nurse-workload constraints: {len(cons_total_nurse_workload)}\n')

    # Total theater capacity
    cons_theater = {}
    for d in instance.days:
      cons_theater[d] = model.addConstr(
        quicksum( p_data.surgery_duration * model._var_patient_day[p,d] for p,p_data in instance.patients.items() )
        <= quicksum( key[1] * var for key,var in model._var_theater_opening.items() if key[0] == d ),
        name=f'TheaterTotalCapacity_on_{d}')
#    sys.stderr.write(f'ADM{self._label}: #theater-opening constraints: {len(cons_theater)}\n')

    objective = 0.0
    objective += quicksum( instance.weights['unscheduled_optional'] * model._var_patient_unscheduled[p] for p in instance.patients )
    objective += quicksum( instance.weights['patient_delay'] * (d - p_data.surgery_release_day) * model._var_patient_day[p,d] for p,p_data in instance.patients.items() for d in instance.days )
    objective += quicksum( instance.weights['nurse_eccessive_workload'] * model._var_shift_workload_excess[s] for s in instance.shifts )
    objective += quicksum( cost * model._var_patient_day[key] for key,cost in self._patients_days_care_bound.items() )
    objective += instance.weights['open_operating_theater'] * quicksum( model._var_theater_opening.values() )
    model.setObjective(objective)

    if self._initial_solution is not None:
      for p in instance.patients:
        sol = self._initial_solution
        d = sol.patients_day.get(p, None)
        if d is None:
          model._var_patient_unscheduled[p].start = 1
        elif not isFixedVar(model._var_patient_unscheduled[p]):
          model._var_patient_unscheduled[p].start = 0

      for key,var in model._var_patient_day.items():
        if isFixedVar(var):
          continue

        p,d = key
        d_adm = sol.patients_day.get(p, None)
        var.start = 1 if d == d_adm else 0

    def solverCallback(model, where):
      if self._interrupted:
        model.terminate()

      instance = model._instance
      if where == GRB.Callback.MIPSOL:
        objectiveValue = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        lower_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        keys = [ key for key,var in model._var_patient_day.items() if not isFixedVar(var) ]
        vals = model.cbGetSolution( [ var for var in model._var_patient_day.values() if not isFixedVar(var) ] )
        patients_day = dict( [ key for key,val in zip(keys, vals) if val > 0.5 ] )
        keys = [ key for key,var in model._var_shift_workload_excess.items() if not isFixedVar(var) ]
        vals = model.cbGetSolution( [ var for var in model._var_shift_workload_excess.values() if not isFixedVar(var) ] )
        shifts_workload = dict( [ (key,int(val + 0.5)) for key,val in zip(keys, vals) if val > 0.01 ] )
        keys = [ key for key,var in model._var_theater_opening.items() if not isFixedVar(var) ]
        vals = model.cbGetSolution( [ var for var in model._var_theater_opening.values() if not isFixedVar(var) ] )
        shifts_theater = dict( [ (key,int(val + 0.5)) for key,val in zip(keys, vals) if val > 0.01 ] )

        solution = self._algo.create_admission_solution(self._room_capacity_reduction, patients_day, self._patients_days_care_bound)
        print(f'ADM{self._label}: {solution}; dual bound: {lower_bound}', flush=True)

      elif where == GRB.Callback.MIP:
        lower_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
      elif where == GRB.Callback.MIPNODE:
        lower_bound = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
      else:
        lower_bound = None

      if lower_bound is not None and self._room_capacity_reduction == 0:
        if lower_bound > self._algo._best_lower_bound:
          self._algo._best_lower_bound = lower_bound
          print(f'Updating global lower bound to {lower_bound}.', flush=True)

    model.optimize(solverCallback)

    print(f'ADM{self._label}: Solving stopped.', flush=True)

