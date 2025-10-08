import sys
import time
import threading
import time
import random
import copy
import math
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




class NursesThread(threading.Thread):

  def __init__(self, instance, algo, time_limit, heuristic_local, heuristic_init, *arg, **kwargs):
    self._instance = instance
    self._algo = algo
    self._time_limit = time_limit
    self._waiting_for_room_solutions = True
    self._current_room_solution = None
    self._best_objective_value = float('inf')
    self._best_room_shift_nurse = None
    self._num_proposed_solutions = 0
    self._heuristic_local = heuristic_local
    self._heuristic_init = heuristic_init
    super().__init__(*arg, **kwargs)

  def signalNoMoreRoomSolutions(self):
    self._waiting_for_room_solutions = False

  def event_shift_min_cost(self, shift_min_cost):
    self._current_room_solution.set_shift_min_cost(shift_min_cost)

  def event_solution(self, room_shift_nurse):
    '''
    Returns the costs of this room/shift-nurse assignment and updates the incumbent.
    '''
    instance = self._instance
    sol = self._current_room_solution

    # Excess workload.
    nurse_shift_capacity = { (n,s):cap for n,n_data in instance.nurses.items() for s,cap in n_data.shifts.items() }
    nurse_shift_work = { key:0 for key in nurse_shift_capacity }

    # Skill level
    num_nurse_skill_deviations = 0
    for key,n in room_shift_nurse.items():
      r,s = key
      d = instance.shift_to_day(s)
      for g in sol.room_day_guests[r,d]:
        g_data = instance.guests[g]
        required = g_data.skill_level_required[s - 3*sol.guests_day[g]]
        num_nurse_skill_deviations += max(0, required - instance.nurses[n].skill_level)

    # Continuity of care.
    num_nurse_continuity_of_care = 0
    for g,r in sol.guests_room.items():
      nurses = set()
      for d in sol.guest_stay(g):
        for s in [3*d, 3*d+1, 3*d+2]:
          n = room_shift_nurse[r,s]
          nurses.add(n)
          nurse_shift_work[n,s] += instance.guests[g].workload_produced[s - 3*sol.guests_day[g]]
      num_nurse_continuity_of_care += len(nurses)

    num_nurse_workload = 0
    for key,work in nurse_shift_work.items():
      cap = nurse_shift_capacity[key]
      num_nurse_workload += max(0, work - cap)

    objective_value = sol.costs_unscheduled + sol.costs_delays + sol.costs_open_theaters + sol.costs_surgeon_transfers + sol.costs_agemix
    objective_value += instance.weights['nurse_eccessive_workload'] * num_nurse_workload
    objective_value += instance.weights['room_nurse_skill'] * num_nurse_skill_deviations
    objective_value += instance.weights['continuity_of_care'] * num_nurse_continuity_of_care

    self._num_proposed_solutions += 1

    if objective_value < self._best_objective_value:
      self._best_objective_value = objective_value
      self._best_room_shift_nurse = room_shift_nurse.copy()
      sol.set_room_shift_nurse(self._best_room_shift_nurse, self._algo.current_time())
      self._algo.event_new_solution(sol)

    return objective_value





  def solve_with_lazy_guest(self, room_sol):
    instance = self._instance

    env = Env("", params = {
      'outputflag': 0,
      'timeLimit': min(self._time_limit, self._algo.remaining_time_parallel()),
      'threads': 1,
      })
    model = Model("Nurses", env=env)

    # Whether room/shift has this nurse.
    model._var_room_shift_nurse = {}
    for n,n_data in instance.nurses.items():
      for s in n_data.shifts:
        d = instance.shift_to_day(s)
        for r in instance.rooms:
          if (r,d) in room_sol.room_day_guests:
            sum_skill_deviation = 0
            for g in room_sol.room_day_guests[r,d]:
              g_data = instance.guests[g]
              rel_s = s - 3 * room_sol.guests_day[g]
              g_skill_level = g_data.skill_level_required[rel_s]
              sum_skill_deviation += max(0, g_skill_level - n_data.skill_level)
            model._var_room_shift_nurse[r,s,n] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['room_nurse_skill'] * sum_skill_deviation, name=f'RoomShiftNurse_{r}_{s}_{n}')
          else:
            model._var_room_shift_nurse[r,s,n] = 0
    print(f'#room-shift-nurse: {len(model._var_room_shift_nurse)}')

    # Number of nurses per guest.
    model._var_guest_nurses = {}
    for g in room_sol.guests_room:
      model._var_guest_nurses[g] = model.addVar(vtype=GRB.CONTINUOUS, lb=3.0, obj=instance.weights['continuity_of_care'], name=f'GuestNurses_{g}')
    print(f'#guest vars: {len(model._var_guest_nurses)}')

    # Excess workload variables.
    model._var_nurse_shift_excess = {}
    for n,n_data in instance.nurses.items():
      for s in n_data.shifts:
        model._var_nurse_shift_excess[n,s] = model.addVar(vtype=GRB.CONTINUOUS, obj=instance.weights['nurse_eccessive_workload'], name=f'NurseShiftExcess_{n}_at_{s}')
    print(f'#excess workload vars: {len(model._var_nurse_shift_excess)}')

    model.update()

    # Each room needs a nurse.
    cons_room_has_nurse = {}
    for r in instance.rooms:
      for s in instance.shifts:
        d = instance.shift_to_day(s)
        gs = room_sol.room_day_guests.get((r,d), [])
        if gs:
          cons_room_has_nurse[r,s] = model.addConstr(
            quicksum( model._var_room_shift_nurse[r,s,n] for n,n_data in instance.nurses.items() if s in n_data.shifts ) == 1,
            name=f'RoomHasNurse_{r}_at_{s}')
    print(f'#room-has-nurse cons: {len(cons_room_has_nurse)}')

    # Nurse workload
    cons_workload = {}
    for n,n_data in instance.nurses.items():
      for s,load in n_data.shifts.items():
        room_workload = { r: 0 for r in instance.rooms }
        d = instance.shift_to_day(s)
        for r in instance.rooms:
          for g in room_sol.room_day_guests.get((r,d), []):
            workload_produced = instance.guests[g].workload_produced[ s - 3 * room_sol.guests_day[g] ]
            room_workload[r] += workload_produced

          cons_workload[n,s] = model.addConstr(
            quicksum( room_workload[r] * model._var_room_shift_nurse.get((r,s,n), 0.0) for r in instance.rooms )
              <= load + model._var_nurse_shift_excess[n,s],
            name=f'NurseCapacity_{n}_{s}')
    print(f'#workload cons: {len(cons_workload)}')

    # Warmstart in case a solution exists.
    if room_sol.has_nurses:
      for r in instance.rooms:
        for s in instance.shifts:
          for n in instance.nurses:
            var = model._var_room_shift_nurse.get((r,s,n), None)
            if var is not None:
              var.start = 1 if room_sol.room_shift_nurse.get((r,s), None) == n else 0

    def solver_callback(model, where):

      # Extract solution.

      if where == GRB.Callback.MIPSOL:
        keys = [ key for key,var in model._var_guest_nurses.items() if not isFixedVar(var) ]
        vals = model.cbGetSolution( [ var for var in model._var_guest_nurses.values() if not isFixedVar(var) ] )
        val_guest_nurses = { key:val for key,val in zip(keys, vals) if val > 0.01 }

        room_shift_nurse_keys = [ key for key,var in model._var_room_shift_nurse.items() if not isFixedVar(var) ]
        room_shift_nurse_vars = [ var for var in model._var_room_shift_nurse.values() if not isFixedVar(var) ]
        room_shift_nurse_vals = model.cbGetSolution( room_shift_nurse_vars )
        val_room_shift_nurse = { key[:3]:val for key,val in zip(room_shift_nurse_keys, room_shift_nurse_vals) if val > 0.01 }
      elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
        keys = [ key for key,var in model._var_guest_nurses.items() if not isFixedVar(var) ]
        vals = model.cbGetNodeRel( [ var for var in model._var_guest_nurses.values() if not isFixedVar(var) ] )
        val_guest_nurses = { key:val for key,val in zip(keys, vals) if val > 0.01 }

        room_shift_nurse_keys = [ key for key,var in model._var_room_shift_nurse.items() if not isFixedVar(var) ]
        room_shift_nurse_vars = [ var for var in model._var_room_shift_nurse.values() if not isFixedVar(var) ]
        room_shift_nurse_vals = model.cbGetNodeRel( room_shift_nurse_vars )
        val_room_shift_nurse = { key[:3]:val for key,val in zip(room_shift_nurse_keys, room_shift_nurse_vals) if val > 0.01 }
      else:
        val_room_shift_nurse = None
        val_guest_nurses = None

      if val_room_shift_nurse is not None:
        for s in instance.shifts:
          all_integer = True
          foo = {}
          for r in instance.rooms:
            for n in instance.nurses:
              val = val_room_shift_nurse.get((r,s,n), 0.0)
              if val > 0.01:
                foo[n,r] = val
                if val < 0.99:
                  all_integer = False
          if not all_integer:
            print(f'Shift {s}: {foo}')
#        print(f'Callback: {val_room_shift_nurse}, {val_guest_nurses}')

        # Generate Benders' optimality cuts.

        guest_nurse_max = { g: {} for g in instance.guests }
        for key,val in val_room_shift_nurse.items():
          r,s,n = key
          d = instance.shift_to_day(s)
          gs = room_sol.room_day_guests.get((r,d), [])
          for g in gs:
            nurse_max = guest_nurse_max[g]
            prev = nurse_max.get(n, (0.0, None))
            if val > prev[0]:
              nurse_max[n] = (val, key)
        num_cuts = 0
        sol_guest_nurses_vars = []
        sol_guest_nurses_vals = []
        total = 0.0
        for g,nurse_max in guest_nurse_max.items():
          estimate = val_guest_nurses.get(g, 0.0)
          real = 0.0
          for data in nurse_max.values():
            real += data[0]
          total += real
          if real > estimate + 0.9:
            model.cbLazy( model._var_guest_nurses[g] >= quicksum( model._var_room_shift_nurse[data[1]] for data in nurse_max.values() ) )
            num_cuts += 1
          if g in model._var_guest_nurses:
            sol_guest_nurses_vars.append( model._var_guest_nurses[g] )
            sol_guest_nurses_vals.append( real )

        if num_cuts:
#          print(f'Generated {num_cuts} cuts. Constructing a primal solution.')

          # Round to a feasible solution.

          room_shift_nurse_max = { }
          for key,val in val_room_shift_nurse.items():
            r,s,n = key
            prev = room_shift_nurse_max.get((r,s), (0.0, None))
            if val > prev[0]:
              room_shift_nurse_max[r,s] = (val, n)

          # Compute excess and coc values.
          nurse_shift_capacity = { (n,s):cap for n,n_data in instance.nurses.items() for s,cap in n_data.shifts.items() }
          nurse_shift_work = { key:0 for key in nurse_shift_capacity }
          guest_num_nurses = { }
          for g,r in room_sol.guests_room.items():
            nurses = set()
            for d in room_sol.guest_stay(g):
              for s in [3*d, 3*d+1, 3*d+2]:
                n = room_shift_nurse_max[r,s][1]
                nurses.add(n)
                nurse_shift_work[n,s] += instance.guests[g].workload_produced[s - 3*room_sol.guests_day[g]]
            guest_num_nurses[g] = len(nurses)

          sol_vars = []
          sol_vals = []
          room_shift_nurse = {}
          for key,data in room_shift_nurse_max.items():
            r,s = key[0],key[1]
            sol_vars.append(model._var_room_shift_nurse[r,s,data[1]])
            sol_vals.append(1.0)
            others = [ model._var_room_shift_nurse[r,s,n] for n in instance.nurses if n != data[1] and (r,s,n) in model._var_room_shift_nurse ]
            sol_vars.extend(others)
            sol_vals.extend([0.0] * len(others))
            room_shift_nurse[r,s] = data[1]
          for g in room_sol.guests_room:
            sol_vars.append( model._var_guest_nurses[g] )
            sol_vals.append( guest_num_nurses[g] )
          for key,cap in nurse_shift_capacity.items():
            work = nurse_shift_work[key]
            sol_vars.append( model._var_nurse_shift_excess[key] )
            sol_vals.append( max(0.0, work - cap) )
          model.cbSetSolution( sol_vars, sol_vals )
          model.cbUseSolution()
#          print(f'Called cbUseSolution.', flush=True)
          self.event_solution(room_shift_nurse)

    model.params.lazyConstraints = 1
    model.optimize( solver_callback )

    if not model.solCount:
      return None

    room_shift_nurse = {}
    for r in instance.rooms:
      for s in instance.shifts:
        for n in instance.nurses:
          var = model._var_room_shift_nurse.get((r,s,n), None)
          if var is not None and var.x > 0.5:
            assert (r,s) not in room_shift_nurse
            room_shift_nurse[r,s] = n
    self.event_solution(room_shift_nurse)

    return model.objBound



  def solve_with_guestwisecare(self, room_sol, with_disaggregation):
    instance = self._instance

    env = Env("", params = {
      'outputflag': 0,
      'timeLimit': min(self._time_limit, self._algo.remaining_time_parallel()),
      'threads': 1,
      'method': 2,
      })
    model = Model("Nurses", env=env)

    model._var_constant = model.addVar(lb=1, ub=1, obj=room_sol.costs_no_nurse)

    # Whether room/shift has this nurse.
    model._var_room_shift_nurse = {}
    for n,n_data in instance.nurses.items():
      for s in n_data.shifts:
        d = instance.shift_to_day(s)
        for r in instance.rooms:
          if (r,d) in room_sol.room_day_guests:
            sum_skill_deviation = 0
            for g in room_sol.room_day_guests[r,d]:
              g_data = instance.guests[g]
              rel_s = s - 3 * room_sol.guests_day[g]
              g_skill_level = g_data.skill_level_required[rel_s]
              sum_skill_deviation += max(0, g_skill_level - n_data.skill_level)
            model._var_room_shift_nurse[r,s,n] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['room_nurse_skill'] * sum_skill_deviation, name=f'RoomShiftNurse_{r}_{s}_{n}')
          else:
            model._var_room_shift_nurse[r,s,n] = 0
#    print(f'#room-shift-nurse: {len(model._var_room_shift_nurse)}')

    # Whether a nurse serves a guest.
    model._var_nurse_guest = {}
    for n in instance.nurses:
      for g in room_sol.guests_room:
        model._var_nurse_guest[n,g] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['continuity_of_care'], name=f'NurseGuest_{n}_{g}')
#    print(f'#nurse-guest vars: {len(model._var_nurse_guest)}')

    # Excess workload variables.
    model._var_nurse_shift_excess = {}
    for n,n_data in instance.nurses.items():
      for s in n_data.shifts:
        model._var_nurse_shift_excess[n,s] = model.addVar(vtype=GRB.CONTINUOUS, obj=instance.weights['nurse_eccessive_workload'], name=f'NurseShiftExcess_{n}_at_{s}')

    model.update()

    # Each room needs a nurse.
    cons_room_has_nurse = {}
    for r in instance.rooms:
      for s in instance.shifts:
        d = instance.shift_to_day(s)
        gs = room_sol.room_day_guests.get((r,d), [])
        if gs:
          cons_room_has_nurse[r,s] = model.addConstr(
            quicksum( model._var_room_shift_nurse[r,s,n] for n,n_data in instance.nurses.items() if s in n_data.shifts ) == 1,
            name=f'RoomHasNurse_{r}_at_{s}')
#    print(f'#room-has-nurse: {len(cons_room_has_nurse)}')

    # Nurse in room implies that nurse cares about patients.
    cons_nurse_guest = {}
    for n in instance.nurses:
      for g,r in room_sol.guests_room.items():
        lhs = []
        g_data = instance.guests[g]
        d_first = room_sol.guests_day[g]
        d_last = min(instance.last_day, d_first + g_data.length_of_stay - 1)
        for s in range(3*d_first, 3*(d_last + 1)):
          if (r,s,n) in model._var_room_shift_nurse and not isinstance(model._var_room_shift_nurse[r,s,n], int):
            lhs.append( model._var_room_shift_nurse[r,s,n] )
        cons_nurse_guest[n,g] = model.addConstr( quicksum( lhs ) <= len(lhs) * model._var_nurse_guest[n,g] )
#    print(f'#aggregated implications: {len(cons_nurse_guest)}')

    # Disaggregation: nurse does not care about guest g => nurse is not there in any shift.
    cons_nurse_guest_disagg = {}
    if with_disaggregation:
      for n,n_data in instance.nurses.items():
        for g,r in room_sol.guests_room.items():
          for s in n_data.shifts:
            if instance.shift_to_day(s) in room_sol.guest_stay(g):
              var = model._var_room_shift_nurse.get((r,s,n), None)
              if var is not None and not isinstance(var, int):
                cons_nurse_guest_disagg[n,g,s] = model.addConstr( var <= model._var_nurse_guest[n,g], f'disagg#n#g#s')

    # Choose at least one nurse among the possible ones for each shift.
    cons_guest_shift_nurse_cover = {}
    for g,r in room_sol.guests_room.items():
      g_data = instance.guests[g]
      d_first = room_sol.guests_day[g]
      d_last = min(instance.last_day, d_first + g_data.length_of_stay - 1)
      for s in range(3*d_first, 3*(d_last + 1)):
        possible_nurses = []
        for n,n_data in instance.nurses.items():
          if s in n_data.shifts:
            possible_nurses.append(n)
#        print(f'Guest {g} in room {r} at shift {s} has these nurses: {possible_nurses}')
        cons_guest_shift_nurse_cover[g,s] = model.addConstr( quicksum( model._var_nurse_guest[n,g] for n in possible_nurses ) >= 1 )
#    print(f'#guest-shift-nurse-cover: {len(cons_guest_shift_nurse_cover)}')

    # Nurse workload
    cons_workload = {}
    for n,n_data in instance.nurses.items():
      for s,load in n_data.shifts.items():
        room_workload = { r: 0 for r in instance.rooms }
        d = instance.shift_to_day(s)
        for r in instance.rooms:
          for g in room_sol.room_day_guests.get((r,d), []):
            workload_produced = instance.guests[g].workload_produced[ s - 3 * room_sol.guests_day[g] ]
            room_workload[r] += workload_produced

          cons_workload[n,s] = model.addConstr(
            quicksum( room_workload[r] * model._var_room_shift_nurse.get((r,s,n), 0.0) for r in instance.rooms )
              <= load + model._var_nurse_shift_excess[n,s],
            name=f'NurseCapacity_{n}_{s}')
#    print(f'#workload: {len(cons_workload)}')

    # Shift minimum costs (objective cut per shift).
    cons_shift_min_cost = {}
    if room_sol.has_shift_min_cost:
      for s in instance.shifts:
        lhs = quicksum( var.obj * var for key,var in model._var_room_shift_nurse.items() if not isFixedVar(var) and key[1] == s and var.obj != 0 ) + \
          quicksum( var.obj * var for key,var in model._var_nurse_shift_excess.items() if not isFixedVar(var) and key[1] == s and var.obj != 0 )
        cons_shift_min_cost[s] = model.addConstr( lhs >= room_sol.shift_min_cost[s], name=f'ShiftMinCosts_{s}')

    if room_sol.has_nurses:
      for r in instance.rooms:
        for s in instance.shifts:
          for n in instance.nurses:
            var = model._var_room_shift_nurse.get((r,s,n), None)
            if var is not None:
              var.start = 1 if room_sol.room_shift_nurse.get((r,s), None) == n else 0

    def solver_callback(model, where):

      if where == GRB.Callback.MIPSOL:
        keys = [ key for key,var in model._var_room_shift_nurse.items() if not isFixedVar(var) ]
        vals = model.cbGetSolution( [ var for var in model._var_room_shift_nurse.values() if not isFixedVar(var) ] )
        room_shift_nurse = { key[:2]:key[2] for key,val in zip(keys, vals) if val > 0.5 }
        self.event_solution(room_shift_nurse)

#    model.write('prob-nurses.lp')
#    model.params.threads = 1
#    model.params.lazyConstraints = 1
#    model.params.heuristics = 0.01
#    model.params.cuts = 0
    model.optimize( solver_callback )
#    else:

    if not model.solCount:
      return None

    room_shift_nurse = {}
    for r in instance.rooms:
      for s in instance.shifts:
        for n in instance.nurses:
          var = model._var_room_shift_nurse.get((r,s,n), None)
          if var is not None and var.x > 0.5:
            assert (r,s) not in room_shift_nurse
            room_shift_nurse[r,s] = n
    self.event_solution(room_shift_nurse)

    return model.objBound, model.runtime

  def solve_with_roomwisecare(self, room_sol, with_disaggregation):
    instance = self._instance

    env = Env("", params = {
      'outputflag': 0,
      'timeLimit': min(self._time_limit, self._algo.remaining_time_parallel()),
      'threads': 1,
      })
    model = Model("Nurses", env=env)

#    for g in room_sol.guests_day:
#      r = room_sol.guests_room[g]
#      days = [ d for d in room_sol.guest_stay(g) ]
#      print(f'Guest {g} stays in {r} on days {min(days)}...{max(days)}')

    # Whether room/shift has this nurse.
    model._var_room_shift_nurse = {}
    for n,n_data in instance.nurses.items():
      for s in n_data.shifts:
        d = instance.shift_to_day(s)
        for r in instance.rooms:
          gs = room_sol.room_day_guests.get((r,d), [])
          if gs:
#            print(f'Considering nurse {n} at shift {s} (day {d}) in room {r}. Guests are {room_sol.room_day_guests.get((r,d), [])}')
            sum_skill_deviation = 0
            for g in room_sol.room_day_guests[r,d]:
              g_data = instance.guests[g]
              rel_s = s - 3 * room_sol.guests_day[g]
              g_skill_level = g_data.skill_level_required[rel_s]
              sum_skill_deviation += max(0, g_skill_level - n_data.skill_level)
            model._var_room_shift_nurse[r,s,n] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['room_nurse_skill'] * sum_skill_deviation, name=f'RoomShiftNurse_{r}_{s}_{n}')
          else:
            model._var_room_shift_nurse[r,s,n] = 0
#    print(f'#room-shift-nurse: {len(model._var_room_shift_nurse)}')

    # How many guests are in a room?
    room_guests = { r:set() for r in instance.rooms }
    for g,r in room_sol.guests_room.items():
      room_guests[r].add(g)

    # Whether a nurse serves a guest.
    model._var_nurse_room = {}
    for n in instance.nurses:
      for r in instance.rooms:
#        room_factor = len(room_guests[r])
        room_factor = 1 #instance.rooms[r].capacity
        model._var_nurse_room[n,r] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['continuity_of_care'] * room_factor, name=f'NurseRoom_{n}_{r}')
#    print(f'#nurse-room vars: {len(model._var_nurse_guest)}')

    # Excess workload variables.
    model._var_nurse_shift_excess = {}
    for n,n_data in instance.nurses.items():
      for s in n_data.shifts:
        model._var_nurse_shift_excess[n,s] = model.addVar(vtype=GRB.CONTINUOUS, obj=instance.weights['nurse_eccessive_workload'], name=f'NurseShiftExcess_{n}_at_{s}')

    model.update()

    # Each room needs a nurse.
    cons_room_has_nurse = {}
    for r in instance.rooms:
      for s in instance.shifts:
        d = instance.shift_to_day(s)
        gs = room_sol.room_day_guests.get((r,d), [])
        if gs:
          cons_room_has_nurse[r,s] = model.addConstr(
            quicksum( model._var_room_shift_nurse[r,s,n] for n,n_data in instance.nurses.items() if s in n_data.shifts ) == 1,
            name=f'RoomHasNurse_{r}_at_{s}')
#    print(f'#room-has-nurse: {len(cons_room_has_nurse)}')

    # Aggregation: nurse does not care about a room => nurse is not there in any shift.
    cons_nurse_room_agg = {}
    for n,n_data in instance.nurses.items():
      for r in instance.rooms:
        lhs = []
        for s in n_data.shifts:
          var = model._var_room_shift_nurse.get((r,s,n), None)
          if var is not None:
            lhs.append( var )
        cons_nurse_room_agg[n,r] = model.addConstr( quicksum( lhs ) <= len(lhs) * model._var_nurse_room[n,r] )
#    print(f'#aggregated implications: {len(cons_nurse_guest)}')

    # Disaggregation: nurse does not care about room r => nurse is not there in any shift.
    cons_nurse_room_disagg = {}
    if with_disaggregation:
      for n,n_data in instance.nurses.items():
        for r in instance.rooms:
          for s in n_data.shifts:
            var = model._var_room_shift_nurse.get((r,s,n), None)
            if var is not None:
              cons_nurse_room_disagg[n,r,s] = model.addConstr( var <= model._var_nurse_room[n,r] )

    # Choose at least one nurse among the possible ones for each shift.
#    cons_guest_shift_nurse_cover = {}
#    for g,r in room_sol.guests_room.items():
#      g_data = instance.guests[g]
#      d_first = room_sol.guests_day[g]
#      d_last = min(instance.last_day, d_first + g_data.length_of_stay - 1)
#      for s in range(3*d_first, 3*(d_last + 1)):
#        possible_nurses = []
#        for n,n_data in instance.nurses.items():
#          if s in n_data.shifts:
#            possible_nurses.append(n)
#        print(f'Guest {g} in room {r} at shift {s} has these nurses: {possible_nurses}')
#        cons_guest_shift_nurse_cover[g,s] = model.addConstr( quicksum( model._var_nurse_guest[n,g] for n in possible_nurses ) >= 1 )
#    print(f'#guest-shift-nurse-cover: {len(cons_guest_shift_nurse_cover)}')

    # Nurse workload
    cons_workload = {}
    for n,n_data in instance.nurses.items():
      for s,load in n_data.shifts.items():
        room_workload = { r: 0 for r in instance.rooms }
        d = instance.shift_to_day(s)
        for r in instance.rooms:
          for g in room_sol.room_day_guests.get((r,d), []):
            workload_produced = instance.guests[g].workload_produced[ s - 3 * room_sol.guests_day[g] ]
            room_workload[r] += workload_produced

        cons_workload[n,s] = model.addConstr(
          quicksum( room_workload[r] * model._var_room_shift_nurse.get((r,s,n), 0.0) for r in instance.rooms )
            <= load + model._var_nurse_shift_excess[n,s],
          name=f'NurseCapacity_{n}_{s}')
#    print(f'#workload: {len(cons_workload)}')


    def solver_callback(model, where):

      if where == GRB.Callback.MIPSOL:
        keys = [ key for key,var in model._var_room_shift_nurse.items() if not isFixedVar(var) ]
        vals = model.cbGetSolution( [ var for var in model._var_room_shift_nurse.values() if not isFixedVar(var) ] )
        room_shift_nurse = { key[:2]:key[2] for key,val in zip(keys, vals) if val > 0.5 }
        self.event_solution(room_shift_nurse)

    if room_sol.has_nurses:
      for r in instance.rooms:
        for s in instance.shifts:
          for n in instance.nurses:
            var = model._var_room_shift_nurse.get((r,s,n), None)
            if var is not None:
              var.start = 1 if room_sol.room_shift_nurse.get((r,s), None) == n else 0

#    model.write('prob-nurses.lp')
    model.params.lazyConstraints = 0
#    model.params.heuristics = 0.01
#    model.params.cuts = 0
    model._room_sol = room_sol
    model.optimize( solver_callback )
#    else:
#    model.optimize()

    if not model.solCount:
      return None

    room_shift_nurse = {}
    for r in instance.rooms:
      for s in instance.shifts:
        for n in instance.nurses:
          var = model._var_room_shift_nurse.get((r,s,n), None)
          if var is not None and evalVar(var) > 0.5:
            assert (r,s) not in room_shift_nurse
            room_shift_nurse[r,s] = n
    self.event_solution(room_shift_nurse)

    return model.objBound



  def solve_timewise(self, room_sol):
    instance = self._instance
    start_time = self._algo.current_time()
    max_time = start_time + min(self._algo.remaining_time_parallel(), self._time_limit)

    env = Env("", params = {
      'outputflag': 0,
      'timeLimit': max_time - start_time,
      'threads': 1,
      'heuristics': 0, # tuned
      'cuts': 2, # tuned 
      })

    dual_bound = 0
    for g in room_sol.guests_day:
      dual_bound += len(instance.shift_types) * instance.weights['continuity_of_care']

    guests_nurses = { g:set() for g in room_sol.guests_day }

    guest_new_nurse_weight = 0.5 / len(room_sol.guests_day)

    # Feasible solution that is constructed.
    room_shift_nurse = {}

    # Exact minimum (workload + skill) costs per shift 
    shift_min_cost = {}

#    run_times = []

    for s in instance.shifts:
      d = instance.shift_to_day(s)

      model = Model("Time", env=env)

      nurses = [ n for n,n_data in instance.nurses.items() if s in n_data.shifts ]
#      print(f'Shift-based model for {len(nurses)} nurses in shift {s}.')

      # Room/nurse variables.
      model._var_room_nurse = {}
      for n in nurses:
        n_data = instance.nurses[n]
        for r in instance.rooms:
          gs = room_sol.room_day_guests.get((r,d), [])
          if gs:
            # Sum of skill level deviations
            sum_skill_deviation = 0
            # Number of guests this nurse would be new to.
#            count_new_guest_nurse_pairs = 0
            for g in room_sol.room_day_guests[r,d]:
              g_data = instance.guests[g]
              rel_s = s - 3 * room_sol.guests_day[g]
              g_skill_level = g_data.skill_level_required[rel_s]
              sum_skill_deviation += max(0, g_skill_level - n_data.skill_level)
#              if n not in guests_nurses[g]:
#                count_new_guest_nurse_pairs += 1
            model._var_room_nurse[r,n] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['room_nurse_skill'] * sum_skill_deviation, name=f'RoomNurse_{r}_{n}')
    
      # Excess workload variables.
      model._var_nurse_excess = {}
      for n in nurses:
        model._var_nurse_excess[n] = model.addVar(vtype=GRB.CONTINUOUS, obj=instance.weights['nurse_eccessive_workload'], name=f'NurseShiftExcess_{n}')

      model.update()

      # Each room needs a nurse.
      cons_room_has_nurse = {}
      for r in instance.rooms:
        gs = room_sol.room_day_guests.get((r,d), [])
        if gs:
          cons_room_has_nurse[r] = model.addConstr(
            quicksum( model._var_room_nurse[r,n] for n,n_data in instance.nurses.items() if s in n_data.shifts ) == 1, name=f'RoomHasNurse_{r}')

      # Nurse workload
      cons_workload = {}
      for n in nurses:
        n_data = instance.nurses[n]
        load = n_data.shifts.get(s, 0)
        if load:
          room_workload = { r: 0 for r in instance.rooms }
          for r in instance.rooms:
            for g in room_sol.room_day_guests.get((r,d), []):
              workload_produced = instance.guests[g].workload_produced[ s - 3 * room_sol.guests_day[g] ]
              room_workload[r] += workload_produced
          cons_workload[n] = model.addConstr(
            quicksum( room_workload[r] * model._var_room_nurse.get((r,n), 0.0) for r in instance.rooms )
              <= load + model._var_nurse_excess[n],
            name=f'NurseCapacity_{n}')

      model.optimize()

      shift_min_cost[s] = int(model.objBound + 0.1)

      # Solve again with updated costs.
      for n in nurses:
        n_data = instance.nurses[n]
        for r in instance.rooms:
          gs = room_sol.room_day_guests.get((r,d), [])
          if gs:
            count_new_guest_nurse_pairs = 0
            for g in room_sol.room_day_guests[r,d]:
              if n not in guests_nurses[g]:
                count_new_guest_nurse_pairs += 1
            model._var_room_nurse[r,n].obj += instance.weights['continuity_of_care'] * count_new_guest_nurse_pairs

      for r in instance.rooms:
        for n in nurses:
          n_data = instance.nurses[n]
          if s in n_data.shifts:
            val = evalVar(model._var_room_nurse.get((r,n), 0))
            if val > 0.5:
              assert (r,s) not in room_shift_nurse
              room_shift_nurse[r,s] = n
              for g in room_sol.room_day_guests[r,d]:
                guests_nurses[g].add(n)

    total_time = self._algo.current_time() - start_time

    self.event_shift_min_cost(shift_min_cost)
    primal_value = self.event_solution(room_shift_nurse)

#    print(f'Heuristic solution has objective value {primal_value}. Dual bounds are {shift_min_cost}, totaling {sum(shift_min_cost.values())}')

#    sys.stderr.write(f'Stats: {sorted(run_times)}\n')
#    sys.stderr.flush()

    return dual_bound, total_time










  def run(self):
    instance = self._instance
    algo = self._algo

    first_sleep = True
    while algo.remaining_time_parallel() > 1:
      sol = algo.process_best_room_solution()
      if sol is None:
        if self._waiting_for_room_solutions:
          if first_sleep:
            first_sleep = False
            print(f'NURS: Sleeping.', flush=True)
          time.sleep(0.1)
          continue
        else:
          print(f'NURS: Stopping.', flush=True)
          return

      first_sleep = True
      self._current_room_solution = sol
      self._best_objective_value = float('inf')
      self._best_room_shift_nurse = None
      self._num_proposed_solutions = 0

      # Solve the actual problem.

#      dual_bound2,timewise_time = self.solve_timewise(sol)

#      sys.stderr.write(f'NURS: Quick lower bound is {sol.costs_no_nurse} + {dual_bound2} = {sol.costs_no_nurse+dual_bound2}.\n')
#      sys.stderr.flush()

#      dual_bound = self.solve_with_lazy_guest(sol)

#      old_time_limt = self._time_limit
#      self._time_limit = 0.8 * old_time_limt
#
#      dual_bound = self.solve_with_roomwisecare(sol, True)
#
#      self._time_limit = 0.2 * old_time_limt
#
      heur_started = time.time()

      print(f'NURS: SA for {sol} out of {algo.num_unprocessed_room_solutions + 1}.', flush=True)

      self.solve_with_heuristic_best(sol, self._heuristic_local, sol.room_shift_nurse if self._heuristic_init else None)

      print(f'NURS: SA completed; {time.time() - heur_started:.01f}s.')

#      print(f'NURS: Starting MIP.')
#      dual_bound,guestwise_time = self.solve_with_guestwisecare(sol, False)
#      sol.set_nurses_min_cost(dual_bound)

#      sys.stderr.write(f'NURS: MIP considered {self._num_proposed_solutions} solutions; dual bound: {dual_bound} >= {dual_bound2}; {timewise_time:.1f}s\n')

#      self._time_limit = old_time_limt

#      print(f'NURS: MIP considered {self._num_proposed_solutions} solutions; dual bound: {dual_bound}; {guestwise_time:.1f}s')
      print(f'NURS: Best: {sol}', flush=True)

#      sol.save(f'sol-{sol.costs_total:06d}-{sol.label}.json')

  def solve_with_roomwisecare_changes(self, room_sol):
        instance = self._instance

        env = Env("", params={'outputflag': 0, 'threads': 1, 'timeLimit': min(
            self._time_limit, self._algo.remaining_time_parallel())})
        model = Model("Nurses", env=env)

        wln = model.addVars(instance.nurses, instance.days,
                            lb=0.0, vtype=GRB.CONTINUOUS, name="wln")
        wl = model.addVar(vtype=GRB.CONTINUOUS, name="wl")
        sl = model.addVar(vtype=GRB.CONTINUOUS, name="sl")
        coc = model.addVar(vtype=GRB.CONTINUOUS, name="coc")
        y = model.addVars(instance.nurses, instance.rooms,
                          instance.days, ub=0.0, vtype=GRB.BINARY, name="y")
        delta = model.addVars(instance.nurses, instance.rooms,
                              instance.days, vtype=GRB.BINARY, name="delta")
        skill_obj = model.addVars(instance.nurses, instance.days,
                                  instance.rooms, lb=0.0, vtype=GRB.CONTINUOUS, name="sln")

        nurses_per_shift = {i: [] for i in instance.shifts}

        for n, n_data in instance.nurses.items():
            prev_d = None
            for s in n_data.shifts:
                d = instance.shift_to_day(s)
                workload_produced = {r: 0 for r in instance.rooms}
                nurses_per_shift[s].append(n)

                for r in instance.rooms:
                    y[n, r, d].ub = 1
                    skill_diff = 0
                    for g in room_sol.room_day_guests[r, d]:
                        g_data = instance.guests[g]
                        rel_s = s - 3 * room_sol.guests_day[g]
                        g_skill_level = g_data.skill_level_required[rel_s]
                        skill_diff += max(0, g_skill_level -
                                          n_data.skill_level)
                    model.addConstr(skill_obj[n, d, r] == (
                        skill_diff * y[n, r, d]))

                    if prev_d is not None:
                        model.addConstr(
                            (y[n, r, d] - y[n, r, prev_d]) <= delta[n, r, d])

                    for g in room_sol.room_day_guests.get((r, d), []):
                        workload_produced[r] += instance.guests[g].workload_produced[s -
                                                                                     3 * room_sol.guests_day[g]]

                prev_d = d

                model.addConstr(wln[n, d] >= sum(workload_produced[r] * y[n, r, d]
                                for r in instance.rooms) - n_data.shifts[s], f"Load-N{n}-D{d}-S{s}")

        for r in instance.rooms:
            for s in instance.shifts:
                d = instance.shift_to_day(s)
                model.addConstr(sum(y[n, r, d]
                                for n in nurses_per_shift[s]) == 1)

        model.addConstr(wl == sum(wln[n, d]
                        for n in instance.nurses for d in instance.days))

        model.addConstr(sl == sum(
            skill_obj[n, d, r] for n in instance.nurses for d in instance.days for r in instance.rooms))

        model.addConstr(coc == sum(
            delta[n, r, d] for n in instance.nurses for d in instance.days for r in instance.rooms))

        model.setObjective(instance.weights["continuity_of_care"] * coc +
                           instance.weights["room_nurse_skill"] * sl +
                           instance.weights["nurse_eccessive_workload"] * wl, GRB.MINIMIZE)

        def solver_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                room_shift_nurse = {}
                for r in instance.rooms:
                    for n, n_data in instance.nurses.items():
                        for s in n_data.shifts:
                            d = instance.shift_to_day(s)
                            value = model.cbGetSolution(y[n, r, d])
                            if value > 0.5:
                                room_shift_nurse[r, s] = n

                self.event_solution(room_shift_nurse)

        model.optimize(solver_callback)
        # model.computeIIS()
        # model.write("iis.ilp")

        if not model.solCount:
            return None

        room_shift_nurse = {}
        for r in instance.rooms:
            for n, n_data in instance.nurses.items():
                for s in n_data.shifts:
                    d = instance.shift_to_day(s)
                    if y[n, r, d].X > 0.5:
                        room_shift_nurse[r, s] = n
        self.event_solution(room_shift_nurse)

  def solve_with_heuristic_random(self, room_sol):
        import numpy as np
        import time
        import random
        import copy
        import math
        print("Nurse heuristic")
        instance = self._instance

        unassigned_shifts = {r: list(instance.shifts) for r in instance.rooms}
        room_shift_nurse = {}
        nurses_per_shift = {i: [] for i in instance.shifts}
        workload_per_room_per_shift = {
            r: {s: 0 for s in instance.shifts} for r in instance.rooms}
        skills_per_room_per_shift = {
            r: {s: [] for s in instance.shifts} for r in instance.rooms}
        nurses_per_patient = {p: {n: 0 for n in instance.nurses}
                              for p in instance.guests}
        coc_per_patient = {p: 0 for p in instance.guests}
        skill_diff_per_nurse = {n: 0 for n in instance.nurses}
        workload_per_nurse_per_day = {
            n: {s: 0 for s in instance.shifts} for n in instance.nurses}
        excess_workload_per_nurse = {n: 0 for n in instance.nurses}
        patients_per_room = {r: {d: 0 for d in instance.days}
                             for r in instance.rooms}
        
        for s in instance.shifts:
            d = instance.shift_to_day(s)
            for r in instance.rooms:
                for g in room_sol.room_day_guests[r, d]:
                    g_data = instance.guests[g]
                    rel_s = s - 3 * room_sol.guests_day[g]
                    skills_per_room_per_shift[r][s].append(g_data.skill_level_required[rel_s])
                    workload_per_room_per_shift[r][s] += instance.guests[g].workload_produced[rel_s]
                    patients_per_room[r][d] += 1

        for n, n_data in instance.nurses.items():
            for s in n_data.shifts:
                d = instance.shift_to_day(s)
                nurses_per_shift[s].append(n)
                if s in unassigned_shifts[r]:
                    if workload_per_nurse_per_day[n][s] + workload_per_room_per_shift[r][s] <= n_data.shifts[s]:
                        room_shift_nurse[r, s] = n
                        workload_per_nurse_per_day[n][s] += workload_per_room_per_shift[r][s]
                        unassigned_shifts[r].remove(s)
                        for g in room_sol.room_day_guests[r, d]:
                            nurses_per_patient[g][n] += 1

                        skill_diff_per_nurse[n] += sum(max(0, i - n_data.skill_level) for i in skills_per_room_per_shift[r][s])

        for r in instance.rooms:
            unassigned = list(unassigned_shifts[r])
            for s in unassigned:
                d = instance.shift_to_day(s)

                possible_nurses = list(nurses_per_shift[s])
                possible_nurses.sort(key=lambda n: (
                    workload_per_nurse_per_day[n][s]))
                n = possible_nurses[0]
                n_data = instance.nurses[n]
                room_shift_nurse[r, s] = n
                workload_per_nurse_per_day[n][s] += workload_per_room_per_shift[r][s]
                unassigned_shifts[r].remove(s)
                for g in room_sol.room_day_guests[r, d]:
                    nurses_per_patient[g][n] += 1

                skill_diff_per_nurse[n] += sum(max(0, i - n_data.skill_level)for i in skills_per_room_per_shift[r][s])

        for p in instance.guests:
            coc_per_patient[p] = sum(1 for val in nurses_per_patient[p].values() if val != 0)
            # print(f"{p} {nurses_per_patient[p]} {coc_per_patient[p]}")

        for n, n_data in instance.nurses.items():
            excess_workload_per_nurse[n] = sum(max(0,  workload_per_nurse_per_day[n][s] - n_data.shifts[s]) for s in n_data.shifts)

        # for (r,s) in room_shift_nurse:
        #     nurse = room_shift_nurse[r,s]
        #     print(f"Room {r} Shift {s}: {nurse} ({instance.nurses[nurse].shifts[s]})")

        runtime = min(self._time_limit, self._algo.remaining_time_parallel())
        start_time = time.time()

        best_solution = copy.deepcopy(room_shift_nurse)
        best_objective = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values()) + instance.weights["continuity_of_care"] * sum(
            coc_per_patient.values()) + instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
        
       

        current_solution = copy.deepcopy(room_shift_nurse)


        current_objective = best_objective
        

        new_coc = instance.weights["continuity_of_care"] * sum(coc_per_patient.values()) 
        new_skill = instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
        new_excess = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values()) 
        new_obj = new_coc + new_skill + new_excess
        

        

        temperature = ((1.5*current_objective) -
                       current_objective) / (-math.log(0.5))
        cooling = 0.99
        iterations = 0
        improving_solutions = 0
        iterations_without_improvement = 0
        while True:
            elapsed = time.time() - start_time
           
            if  elapsed >= runtime:
                print(f"STOP SA after {iterations} iterations due to runtime limit {runtime}: Found {improving_solutions} improving solutions.")
                break
                

            if iterations_without_improvement >= 3000:
                print(f"STOP SA after {iterations} iterations due to iterations without improvement {iterations_without_improvement}: Found {improving_solutions} improving solutions.")
                break

            random_shift = random.choice(tuple(instance.shifts))
            random_room = random.choice(tuple(instance.rooms.keys()))

            current_nurse = current_solution[random_room, random_shift]
            avail_nurses = list(nurses_per_shift[random_shift])

            avail_nurses.remove(current_nurse)
            new_nurse = random.sample(avail_nurses, k=1)[0]

            current_nurse_workload = workload_per_nurse_per_day[current_nurse][random_shift] - \
                workload_per_room_per_shift[random_room][random_shift]
            new_nurse_workload = workload_per_nurse_per_day[new_nurse][random_shift] + \
                workload_per_room_per_shift[random_room][random_shift]
            delta_excess_new = max(0, new_nurse_workload - instance.nurses[new_nurse].shifts[random_shift]) - max(
                0, workload_per_nurse_per_day[new_nurse][random_shift] - instance.nurses[new_nurse].shifts[random_shift])
            delta_excess_current = max(0, current_nurse_workload - instance.nurses[current_nurse].shifts[random_shift]) - max(
                0, workload_per_nurse_per_day[current_nurse][random_shift] - instance.nurses[current_nurse].shifts[random_shift])
            delta_excess = delta_excess_new + delta_excess_current


            current_nurse_skill = sum(max(0, i - instance.nurses[current_nurse].skill_level) for i in skills_per_room_per_shift[random_room][random_shift])
            new_nurse_skill = sum(max(0, i - instance.nurses[new_nurse].skill_level) for i in skills_per_room_per_shift[random_room][random_shift])
            delta_skill = new_nurse_skill - current_nurse_skill

            delta_coc = 0
            day = instance.shift_to_day(random_shift)
            for p in room_sol.room_day_guests[random_room, day]:
                counter_current_nurse = nurses_per_patient[p][current_nurse]
                if counter_current_nurse == 1:
                    delta_coc -= 1
                counter_new_nurse = nurses_per_patient[p][new_nurse]
                if counter_new_nurse == 0:
                    delta_coc += 1

            delta_total = instance.weights["nurse_eccessive_workload"] * delta_excess + \
                instance.weights["continuity_of_care"] * delta_coc + \
                instance.weights["room_nurse_skill"] * delta_skill
            
            iterations_without_improvement +=1
            if delta_total < 0 or random.random() < math.exp(-delta_total / temperature):
                current_solution[random_room, random_shift] = new_nurse
                workload_per_nurse_per_day[current_nurse][random_shift] = current_nurse_workload
                workload_per_nurse_per_day[new_nurse][random_shift] = new_nurse_workload
                excess_workload_per_nurse[new_nurse] += delta_excess_new
                excess_workload_per_nurse[current_nurse] += delta_excess_current
                skill_diff_per_nurse[current_nurse] -= current_nurse_skill
                skill_diff_per_nurse[new_nurse] += new_nurse_skill
                for p in room_sol.room_day_guests[random_room, day]:
                    nurses_per_patient[p][new_nurse] += 1
                    nurses_per_patient[p][current_nurse] -= 1
                    coc_per_patient[p] = sum(1 for val in nurses_per_patient[p].values() if val != 0)

                new_coc = instance.weights["continuity_of_care"] * sum(
                    coc_per_patient.values()) 
                new_skill = instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
                new_excess = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values()) 
                new_obj = new_coc + new_skill + new_excess

                current_objective = new_obj
                if current_objective < best_objective:
                    iterations_without_improvement = 0
                    improving_solutions += 1
                    best_objective = current_objective
                    best_solution = copy.deepcopy(current_solution)
                    
                    self.event_solution(best_solution)
                   

            temperature *= cooling
            iterations += 1

        print(f"Obj {new_obj} Ex {new_excess} Skill {new_skill} Coc {new_coc} ")
        print(f"{elapsed} Best solution {current_objective}, before {best_objective} delta {delta_total} diff {best_objective + delta_total}")
        self.event_solution(best_solution)

  def solve_with_heuristic_best(self, room_sol, only_local_search = True, init_nurse_sol = None):

    instance = self._instance
    room_shift_nurse = {}
    nurses_per_shift = {i: [] for i in instance.shifts}
    workload_per_room_per_shift = {
        r: {s: 0 for s in instance.shifts} for r in instance.rooms}
    skills_per_room_per_shift = {
        r: {s: [] for s in instance.shifts} for r in instance.rooms}
    nurses_per_patient = {p: {n: 0 for n in instance.nurses}
                          for p in instance.guests}
    coc_per_patient = {p: 0 for p in instance.guests}
    skill_diff_per_nurse = {n: 0 for n in instance.nurses}
    workload_per_nurse_per_day = {
        n: {s: 0 for s in instance.shifts} for n in instance.nurses}
    excess_workload_per_nurse = {n: 0 for n in instance.nurses}
    patients_per_room = {r: {d: 0 for d in instance.days}
                        for r in instance.rooms}

    for s in instance.shifts:
        d = instance.shift_to_day(s)
        for r in instance.rooms:
            for g in room_sol.room_day_guests[r, d]:
                g_data = instance.guests[g]
                rel_s = s - 3 * room_sol.guests_day[g]
                skills_per_room_per_shift[r][s].append(g_data.skill_level_required[rel_s])
                workload_per_room_per_shift[r][s] += instance.guests[g].workload_produced[rel_s]
                patients_per_room[r][d] += 1

    unassigned_shifts = {r: list(instance.shifts) for r in instance.rooms}
    if init_nurse_sol is None:
      for n, n_data in instance.nurses.items():
          for s in n_data.shifts:
              d = instance.shift_to_day(s)
              nurses_per_shift[s].append(n)
              if s in unassigned_shifts[r]:
                  if workload_per_nurse_per_day[n][s] + workload_per_room_per_shift[r][s] <= n_data.shifts[s]:
                      room_shift_nurse[r, s] = n
                      workload_per_nurse_per_day[n][s] += workload_per_room_per_shift[r][s]
                      unassigned_shifts[r].remove(s)
                      for g in room_sol.room_day_guests[r, d]:
                          nurses_per_patient[g][n] += 1

                      skill_diff_per_nurse[n] += sum(max(0, i - n_data.skill_level) for i in skills_per_room_per_shift[r][s])
    else:
       for n, n_data in instance.nurses.items():
          for s in n_data.shifts:
              d = instance.shift_to_day(s)
              nurses_per_shift[s].append(n)

    for r in instance.rooms:
        unassigned = list(unassigned_shifts[r])
        for s in unassigned:
            d = instance.shift_to_day(s)

            n = None
            if init_nurse_sol is not None:
              n = init_nurse_sol[r, s]
            if n is None:
              possible_nurses = list(nurses_per_shift[s])
              possible_nurses.sort(key=lambda n: (
                  workload_per_nurse_per_day[n][s]))
              n = possible_nurses[0]

            n_data = instance.nurses[n]
            room_shift_nurse[r, s] = n
            workload_per_nurse_per_day[n][s] += workload_per_room_per_shift[r][s]
            unassigned_shifts[r].remove(s)
            for g in room_sol.room_day_guests[r, d]:
                nurses_per_patient[g][n] += 1

            skill_diff_per_nurse[n] += sum(max(0, i - n_data.skill_level)for i in skills_per_room_per_shift[r][s])

    for p in instance.guests:
        coc_per_patient[p] = sum(1 for val in nurses_per_patient[p].values() if val != 0)
        # print(f"{p} {nurses_per_patient[p]} {coc_per_patient[p]}")

    for n, n_data in instance.nurses.items():
        excess_workload_per_nurse[n] = sum(max(0,  workload_per_nurse_per_day[n][s] - n_data.shifts[s]) for s in n_data.shifts)

    # for (r,s) in room_shift_nurse:
    #     nurse = room_shift_nurse[r,s]
    #     print(f"Room {r} Shift {s}: {nurse} ({instance.nurses[nurse].shifts[s]})")

    runtime = min(self._time_limit, self._algo.remaining_time_parallel()-2)
    start_time = time.time()

    best_solution = copy.deepcopy(room_shift_nurse)
    best_objective = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values()) + instance.weights["continuity_of_care"] * sum(
        coc_per_patient.values()) + instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())



    current_solution = copy.deepcopy(room_shift_nurse)


    current_objective = best_objective


    new_coc = instance.weights["continuity_of_care"] * sum(coc_per_patient.values())
    new_skill = instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
    new_excess = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values())
    new_obj = new_coc + new_skill + new_excess




    temperature = ((1.05*current_objective) -
                  current_objective) / (-math.log(0.5))
    cooling = 0.999



    iterations = 0
    improving_solutions = 0
    iterations_without_improvement = 0
    while True:
        elapsed = time.time() - start_time

        if  elapsed >= runtime:
            #print(f"STOP SA after {iterations} iterations due to runtime limit {runtime}: Found {improving_solutions} improving solutions. {elapsed}s {iterations_without_improvement} iterations without improvement {self._algo.remaining_time_parallel()}")
            break


        if iterations_without_improvement >= 5000:
            #print(f"STOP SA after {iterations} iterations due to iterations without improvement {iterations_without_improvement}: Found {improving_solutions} improving solutions. {elapsed}s {iterations_without_improvement} iterations without improvement {self._algo.remaining_time_parallel()}")
            break

        random_shift = random.choice(tuple(instance.shifts))
        random_room = random.choice(tuple(instance.rooms.keys()))

        current_nurse = current_solution[random_room, random_shift]
        avail_nurses = list(nurses_per_shift[random_shift])

        avail_nurses.remove(current_nurse)
        
        # if no nurses to switch out nurse, abort iteration and return to beginning to select a new room-shift combination
        if len(avail_nurses) < 1:
          continue
          
          

        best_delta = {}
        best_current_nurse_workload = {}
        best_new_nurse_workload = {}
        best_delta_excess_new = {}
        best_delta_excess_current = {}
        best_current_nurse_skill = {}
        best_new_nurse_skill = {}
        for new_nurse in avail_nurses:

            current_nurse_workload = workload_per_nurse_per_day[current_nurse][random_shift] - \
                workload_per_room_per_shift[random_room][random_shift]
            new_nurse_workload = workload_per_nurse_per_day[new_nurse][random_shift] + \
                workload_per_room_per_shift[random_room][random_shift]
            delta_excess_new = max(0, new_nurse_workload - instance.nurses[new_nurse].shifts[random_shift]) - max(
                0, workload_per_nurse_per_day[new_nurse][random_shift] - instance.nurses[new_nurse].shifts[random_shift])
            delta_excess_current = max(0, current_nurse_workload - instance.nurses[current_nurse].shifts[random_shift]) - max(
                0, workload_per_nurse_per_day[current_nurse][random_shift] - instance.nurses[current_nurse].shifts[random_shift])
            delta_excess = delta_excess_new + delta_excess_current


            current_nurse_skill = sum(max(0, i - instance.nurses[current_nurse].skill_level) for i in skills_per_room_per_shift[random_room][random_shift])
            new_nurse_skill = sum(max(0, i - instance.nurses[new_nurse].skill_level) for i in skills_per_room_per_shift[random_room][random_shift])
            delta_skill = new_nurse_skill - current_nurse_skill

            delta_coc = 0
            day = instance.shift_to_day(random_shift)
            for p in room_sol.room_day_guests[random_room, day]:
                counter_current_nurse = nurses_per_patient[p][current_nurse]
                if counter_current_nurse == 1:
                    delta_coc -= 1
                counter_new_nurse = nurses_per_patient[p][new_nurse]
                if counter_new_nurse == 0:
                    delta_coc += 1

            delta_total = instance.weights["nurse_eccessive_workload"] * delta_excess + \
                instance.weights["continuity_of_care"] * delta_coc + \
                instance.weights["room_nurse_skill"] * delta_skill



            best_delta[new_nurse] = delta_total
            best_current_nurse_workload[new_nurse] = current_nurse_workload
            best_new_nurse_workload[new_nurse] = new_nurse_workload
            best_delta_excess_new[new_nurse] = delta_excess_new
            best_delta_excess_current[new_nurse] = delta_excess_current
            best_current_nurse_skill[new_nurse] = current_nurse_skill
            best_new_nurse_skill[new_nurse] = new_nurse_skill

        # Determine min and max for normalization
        min_delta = min(best_delta.values())
        max_delta = max(best_delta.values())
        best_nurse = None
        if min_delta != max_delta:
            
            # Normalize delta values into interval [0.02,1.02] for 1.02 being the best solution and 0.02 being the worst solution
            norm_delta = {}
            avail_nurses.sort(key=lambda n: (best_delta[n]))
            total = 0.0
            for n in avail_nurses:
                norm_delta[n] = (1-((best_delta[n] - min_delta)/(max_delta - min_delta))) + 0.02
                total += norm_delta[n]

            # Select random nurse based on weight from (normalized value / (total of normalized values))
            random_number = random.random()
            cumm_value = 0.0
            for n in avail_nurses:
                value = norm_delta[n]/total
                #print(f" {n} {best_delta[n]} {norm_delta[n]} {value} {cumm_value + value}")
                if random_number >= cumm_value and random_number <= (cumm_value + value):
                    best_nurse  = n
                    break
                cumm_value += value
                

            #print(f"Random {random_number} Choice {best_nurse}")
        else:
            best_nurse = random.choice(avail_nurses)
            #print(f" {min_delta} {max_delta} {best_delta} ")


        #print(f"Choice: {best_nurse}")
        iterations_without_improvement +=1
        accept = False
        if only_local_search:
          if best_delta[best_nurse] < 0:
            accept = True
        else:
          if best_delta[best_nurse] < 0 or random.random() < math.exp(-best_delta[best_nurse] / temperature):
            accept = True

        if accept:
            current_solution[random_room, random_shift] = best_nurse
            workload_per_nurse_per_day[current_nurse][random_shift] = best_current_nurse_workload[best_nurse]
            workload_per_nurse_per_day[best_nurse][random_shift] = best_new_nurse_workload[best_nurse]
            excess_workload_per_nurse[best_nurse] += best_delta_excess_new[best_nurse]
            excess_workload_per_nurse[current_nurse] += best_delta_excess_current[best_nurse]
            skill_diff_per_nurse[current_nurse] -= best_current_nurse_skill[best_nurse]
            skill_diff_per_nurse[best_nurse] += best_new_nurse_skill[best_nurse]
            for p in room_sol.room_day_guests[random_room, day]:
                nurses_per_patient[p][best_nurse] += 1
                nurses_per_patient[p][current_nurse] -= 1
                coc_per_patient[p] = sum(1 for val in nurses_per_patient[p].values() if val != 0)

            new_coc = instance.weights["continuity_of_care"] * sum(
                coc_per_patient.values())
            new_skill = instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
            new_excess = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values())
            new_obj = new_coc + new_skill + new_excess

            current_objective = new_obj
            if current_objective < best_objective:
                iterations_without_improvement = 0
                improving_solutions += 1
                best_objective = current_objective
                best_solution = copy.deepcopy(current_solution)

                #self.event_solution(best_solution)


        temperature *= cooling
        iterations += 1

      #print(f"Obj {new_obj} Ex {new_excess} Skill {new_skill} Coc {new_coc} ")
      #print(f"{elapsed} Best solution {current_objective}, before {best_objective} delta {delta_total} diff {best_objective + delta_total}")

    self.event_solution(best_solution)

  def solve_with_grasp(self, room_sol):
        import numpy as np
        import time
        import random
        import copy
        import math
        print("Nurse heuristic")
        instance = self._instance

        unassigned_shifts = {r: list(instance.shifts) for r in instance.rooms}
        room_shift_nurse = {}
        nurses_per_shift = {i: [] for i in instance.shifts}
        workload_per_room_per_shift = {
            r: {s: 0 for s in instance.shifts} for r in instance.rooms}
        skills_per_room_per_shift = {
            r: {s: [] for s in instance.shifts} for r in instance.rooms}
        nurses_per_patient = {p: {n: 0 for n in instance.nurses}
                              for p in instance.guests}
        coc_per_patient = {p: 0 for p in instance.guests}
        skill_diff_per_nurse = {n: 0 for n in instance.nurses}
        workload_per_nurse_per_day = {
            n: {s: 0 for s in instance.shifts} for n in instance.nurses}
        excess_workload_per_nurse = {n: 0 for n in instance.nurses}
        patients_per_room = {r: {d: 0 for d in instance.days}
                             for r in instance.rooms}
        
        for s in instance.shifts:
            d = instance.shift_to_day(s)
            for r in instance.rooms:
                for g in room_sol.room_day_guests[r, d]:
                    g_data = instance.guests[g]
                    rel_s = s - 3 * room_sol.guests_day[g]
                    skills_per_room_per_shift[r][s].append(g_data.skill_level_required[rel_s])
                    workload_per_room_per_shift[r][s] += instance.guests[g].workload_produced[rel_s]
                    patients_per_room[r][d] += 1

        
        for n, n_data in instance.nurses.items():
            for s in n_data.shifts:
                d = instance.shift_to_day(s)
                nurses_per_shift[s].append(n)

        

        # for (r,s) in room_shift_nurse:
        #     nurse = room_shift_nurse[r,s]
        #     print(f"Room {r} Shift {s}: {nurse} ({instance.nurses[nurse].shifts[s]})")

        runtime = min(self._time_limit, self._algo.remaining_time_parallel())
        start_time_grasp = time.time()
        iterations_grasp = 0

        while True:
            elapsed_grasp = time.time() - start_time_grasp
           
            if  elapsed_grasp >= runtime:
                print(f"STOP GRASP after {iterations_grasp} iterations due to runtime limit {runtime}:{self._algo.remaining_time_parallel()}")
                break
                
            iterations_grasp +=1
            shuffled_rooms = list(instance.rooms)
            random.shuffle(shuffled_rooms)
            for r in shuffled_rooms:
                unassigned = list(unassigned_shifts[r])
                random.shuffle(unassigned)
                for s in unassigned:
                    d = instance.shift_to_day(s)

                    possible_nurses = list(nurses_per_shift[s])
                    possible_nurses.sort(key=lambda n: (
                        workload_per_nurse_per_day[n][s]))
                    n = possible_nurses[0]
                    n_data = instance.nurses[n]
                    room_shift_nurse[r, s] = n
                    workload_per_nurse_per_day[n][s] += workload_per_room_per_shift[r][s]
                    unassigned_shifts[r].remove(s)
                    for g in room_sol.room_day_guests[r, d]:
                        nurses_per_patient[g][n] += 1

                    skill_diff_per_nurse[n] += sum(max(0, i - n_data.skill_level)for i in skills_per_room_per_shift[r][s])

            for p in instance.guests:
                coc_per_patient[p] = sum(1 for val in nurses_per_patient[p].values() if val != 0)
                # print(f"{p} {nurses_per_patient[p]} {coc_per_patient[p]}")

            for n, n_data in instance.nurses.items():
                excess_workload_per_nurse[n] = sum(max(0,  workload_per_nurse_per_day[n][s] - n_data.shifts[s]) for s in n_data.shifts)


            best_solution = copy.deepcopy(room_shift_nurse)
            best_objective = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values()) + instance.weights["continuity_of_care"] * sum(
                coc_per_patient.values()) + instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
            
        

            current_solution = copy.deepcopy(room_shift_nurse)


            current_objective = best_objective
            

            new_coc = instance.weights["continuity_of_care"] * sum(coc_per_patient.values()) 
            new_skill = instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
            new_excess = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values()) 
            new_obj = new_coc + new_skill + new_excess
            

            
            
            iterations = 0
            improving_solutions = 0
            iterations_without_improvement = 0
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
            
                if  elapsed >= runtime/5:
                    print(f"STOP SA after {iterations} iterations due to runtime limit {runtime}: Found {improving_solutions} improving solutions. {elapsed}s {iterations_without_improvement} iterations without improvement {self._algo.remaining_time_parallel()}")
                    break
                    

                # if iterations_without_improvement >= 1000:
                #     print(f"STOP SA after {iterations} iterations due to iterations without improvement {iterations_without_improvement}: Found {improving_solutions} improving solutions. {elapsed}s {iterations_without_improvement} iterations without improvement {self._algo.remaining_time_parallel()}")
                #     break

                random_shift = random.choice(tuple(instance.shifts))
                random_room = random.choice(tuple(instance.rooms.keys()))

                current_nurse = current_solution[random_room, random_shift]
                avail_nurses = list(nurses_per_shift[random_shift])

                avail_nurses.remove(current_nurse)
                best_delta = {}
                best_current_nurse_workload = {}
                best_new_nurse_workload = {}
                best_delta_excess_new = {}
                best_delta_excess_current = {}
                best_current_nurse_skill = {}
                best_new_nurse_skill = {}
                for new_nurse in avail_nurses:

                    current_nurse_workload = workload_per_nurse_per_day[current_nurse][random_shift] - \
                        workload_per_room_per_shift[random_room][random_shift]
                    new_nurse_workload = workload_per_nurse_per_day[new_nurse][random_shift] + \
                        workload_per_room_per_shift[random_room][random_shift]
                    delta_excess_new = max(0, new_nurse_workload - instance.nurses[new_nurse].shifts[random_shift]) - max(
                        0, workload_per_nurse_per_day[new_nurse][random_shift] - instance.nurses[new_nurse].shifts[random_shift])
                    delta_excess_current = max(0, current_nurse_workload - instance.nurses[current_nurse].shifts[random_shift]) - max(
                        0, workload_per_nurse_per_day[current_nurse][random_shift] - instance.nurses[current_nurse].shifts[random_shift])
                    delta_excess = delta_excess_new + delta_excess_current


                    current_nurse_skill = sum(max(0, i - instance.nurses[current_nurse].skill_level) for i in skills_per_room_per_shift[random_room][random_shift])
                    new_nurse_skill = sum(max(0, i - instance.nurses[new_nurse].skill_level) for i in skills_per_room_per_shift[random_room][random_shift])
                    delta_skill = new_nurse_skill - current_nurse_skill

                    delta_coc = 0
                    day = instance.shift_to_day(random_shift)
                    for p in room_sol.room_day_guests[random_room, day]:
                        counter_current_nurse = nurses_per_patient[p][current_nurse]
                        if counter_current_nurse == 1:
                            delta_coc -= 1
                        counter_new_nurse = nurses_per_patient[p][new_nurse]
                        if counter_new_nurse == 0:
                            delta_coc += 1

                    delta_total = instance.weights["nurse_eccessive_workload"] * delta_excess + \
                        instance.weights["continuity_of_care"] * delta_coc + \
                        instance.weights["room_nurse_skill"] * delta_skill

                    
                    
                    best_delta[new_nurse] = delta_total
                    best_current_nurse_workload[new_nurse] = current_nurse_workload
                    best_new_nurse_workload[new_nurse] = new_nurse_workload
                    best_delta_excess_new[new_nurse] = delta_excess_new
                    best_delta_excess_current[new_nurse] = delta_excess_current
                    best_current_nurse_skill[new_nurse] = current_nurse_skill
                    best_new_nurse_skill[new_nurse] = new_nurse_skill


                min_delta = min(best_delta.values())
                max_delta = max(best_delta.values())

                best_nurse = None

                if min_delta != max_delta:
                    random_number = random.random()

                    avail_nurses.sort(key=lambda n: (best_delta[n]))

                    value = 0.0
                    for n in avail_nurses:
                        value += (best_delta[n] - min_delta)/(max_delta - min_delta) 
                        if random_number <= value:
                            break
                        best_nurse  = n
                        #print(value)

                    #print(f" {min_delta} {max_delta} {best_delta} {avail_nurses} {random_number}")

                else:
                    best_nurse = random.choice(avail_nurses)    
                    #print(f" {min_delta} {max_delta} {best_delta} ")


                #print(f"Choice: {best_nurse}")
                iterations_without_improvement +=1
                if best_delta[best_nurse] < 0:
                    current_solution[random_room, random_shift] = best_nurse
                    workload_per_nurse_per_day[current_nurse][random_shift] = best_current_nurse_workload[best_nurse]
                    workload_per_nurse_per_day[best_nurse][random_shift] = best_new_nurse_workload[best_nurse]
                    excess_workload_per_nurse[best_nurse] += best_delta_excess_new[best_nurse]
                    excess_workload_per_nurse[current_nurse] += best_delta_excess_current[best_nurse]
                    skill_diff_per_nurse[current_nurse] -= best_current_nurse_skill[best_nurse]
                    skill_diff_per_nurse[best_nurse] += best_new_nurse_skill[best_nurse]
                    for p in room_sol.room_day_guests[random_room, day]:
                        nurses_per_patient[p][best_nurse] += 1
                        nurses_per_patient[p][current_nurse] -= 1
                        coc_per_patient[p] = sum(1 for val in nurses_per_patient[p].values() if val != 0)

                    new_coc = instance.weights["continuity_of_care"] * sum(
                        coc_per_patient.values()) 
                    new_skill = instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
                    new_excess = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values()) 
                    new_obj = new_coc + new_skill + new_excess

                    current_objective = new_obj
                    if current_objective < best_objective:
                        iterations_without_improvement = 0
                        improving_solutions += 1
                        best_objective = current_objective
                        best_solution = copy.deepcopy(current_solution)
                        
                        #self.event_solution(best_solution)
                    

            
                iterations += 1

        print(f"Obj {new_obj} Ex {new_excess} Skill {new_skill} Coc {new_coc} ")
        print(f"{elapsed} Best solution {current_objective}, before {best_objective} delta {delta_total} diff {best_objective + delta_total}")
        self.event_solution(best_solution)

  def solve_with_heuristic_track(self, room_sol):


        import numpy as np
        import time
        import random
        import copy
        import math
        print("Nurse heuristic")
        instance = self._instance

        
        unassigned_shifts = {r: list(instance.shifts) for r in instance.rooms}
        room_shift_nurse = {}
        cost = {}
        nurses_per_shift = {i: [] for i in instance.shifts}
        workload_per_room_per_shift = {
            r: {s: 0 for s in instance.shifts} for r in instance.rooms}
        skills_per_room_per_shift = {
            r: {s: [] for s in instance.shifts} for r in instance.rooms}
        nurses_per_patient = {p: {n: 0 for n in instance.nurses}
                              for p in instance.guests}
        coc_per_patient = {p: 0 for p in instance.guests}
        skill_diff_per_nurse = {n: 0 for n in instance.nurses}
        workload_per_nurse_per_day = {
            n: {s: 0 for s in instance.shifts} for n in instance.nurses}
        excess_workload_per_nurse = {n: 0 for n in instance.nurses}
        patients_per_room = {r: {d: 0 for d in instance.days}
                             for r in instance.rooms}
        
        for s in instance.shifts:
            d = instance.shift_to_day(s)
            for r in instance.rooms:
                for g in room_sol.room_day_guests[r, d]:
                    g_data = instance.guests[g]
                    rel_s = s - 3 * room_sol.guests_day[g]
                    skills_per_room_per_shift[r][s].append(g_data.skill_level_required[rel_s])
                    workload_per_room_per_shift[r][s] += instance.guests[g].workload_produced[rel_s]
                    patients_per_room[r][d] += 1

        for n, n_data in instance.nurses.items():
            for s in n_data.shifts:
                d = instance.shift_to_day(s)
                nurses_per_shift[s].append(n)
                if s in unassigned_shifts[r]:
                    if workload_per_nurse_per_day[n][s] + workload_per_room_per_shift[r][s] <= n_data.shifts[s]:
                        room_shift_nurse[r, s] = n
                        workload_per_nurse_per_day[n][s] += workload_per_room_per_shift[r][s]
                        room_skill = sum(max(0, i - n_data.skill_level) for i in skills_per_room_per_shift[r][s])
                        skill_diff_per_nurse[n] += room_skill
                        unassigned_shifts[r].remove(s)
                        coc = 0
                        for g in room_sol.room_day_guests[r, d]:
                            nurses_per_patient[g][n] += 1
                            coc += 1 - (nurses_per_patient[g][n]/(instance.guests[g].length_of_stay*3))


                        cost[n,r,s] = instance.weights["room_nurse_skill"] *room_skill
                        cost[n,r,s] += instance.weights["nurse_eccessive_workload"]*workload_per_room_per_shift[r][s]
                        cost[n,r,s] += instance.weights["continuity_of_care"]  * coc 

        for r in instance.rooms:
            unassigned = list(unassigned_shifts[r])
            for s in unassigned:
                d = instance.shift_to_day(s)

                possible_nurses = list(nurses_per_shift[s])
                possible_nurses.sort(key=lambda n: (
                    workload_per_nurse_per_day[n][s]))
                n = possible_nurses[0]
                n_data = instance.nurses[n]
                room_shift_nurse[r, s] = n
                workload_per_nurse_per_day[n][s] += workload_per_room_per_shift[r][s]
                unassigned_shifts[r].remove(s)
                coc = 0
                for g in room_sol.room_day_guests[r, d]:
                    nurses_per_patient[g][n] += 1
                    coc += 1 - (nurses_per_patient[g][n]/(instance.guests[g].length_of_stay*3))

                room_skill = sum(max(0, i - n_data.skill_level)for i in skills_per_room_per_shift[r][s])
                skill_diff_per_nurse[n] += room_skill

                cost[n,r,s] = instance.weights["room_nurse_skill"] *room_skill
                cost[n,r,s] += instance.weights["nurse_eccessive_workload"]*workload_per_room_per_shift[r][s]
                cost[n,r,s] += instance.weights["continuity_of_care"]  * coc 

        for p in instance.guests:
            coc_per_patient[p] = sum(1 for val in nurses_per_patient[p].values() if val != 0)
            # print(f"{p} {nurses_per_patient[p]} {coc_per_patient[p]}")

        for n, n_data in instance.nurses.items():
            excess_workload_per_nurse[n] = sum(max(0,  workload_per_nurse_per_day[n][s] - n_data.shifts[s]) for s in n_data.shifts)

        # for (r,s) in room_shift_nurse:
        #     nurse = room_shift_nurse[r,s]
        #     print(f"Room {r} Shift {s}: {nurse} ({instance.nurses[nurse].shifts[s]})")

        runtime = min(self._time_limit, self._algo.remaining_time_parallel())
        start_time = time.time()

        best_solution = copy.deepcopy(room_shift_nurse)
        best_objective = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values()) + instance.weights["continuity_of_care"] * sum(
            coc_per_patient.values()) + instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
        
       

        current_solution = copy.deepcopy(room_shift_nurse)


        current_objective = best_objective
        

        new_coc = instance.weights["continuity_of_care"] * sum(coc_per_patient.values()) 
        new_skill = instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
        new_excess = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values()) 
        new_obj = new_coc + new_skill + new_excess
        

        
        self.event_solution(best_solution)
        start_temperature = ((1.5*current_objective) -
                       current_objective) / (-math.log(0.5))
        cooling = 0.99
        iterations = 0
        improving_solutions = 0
        iterations_without_improvement = 0
        temperature = start_temperature
        while True:
            elapsed = time.time() - start_time
           
            if  elapsed >= runtime:
                print(f"STOP SA after {iterations} iterations due to runtime limit {runtime}: Found {improving_solutions} improving solutions.")
                break

         
            if iterations_without_improvement >= 50000:
                print(f"STOP SA after {iterations} iterations due to iterations without improvement {iterations_without_improvement}: Found {improving_solutions} improving solutions.")
                break
            
            non_zero_cost = {key: value for key, value in cost.items() if value > 0}
            selected_elements = random.sample(list(non_zero_cost.items()), k=5)  
            selected_elements.sort(key=lambda n: (n[1]),reverse=True)
            current_nurse, random_room, random_shift = selected_elements[0][0]
            # print(selected_elements)
            # print(selected_elements[0])
            # random_shift = random.choice(tuple(instance.shifts))
            # random_room = random.choice(tuple(instance.rooms.keys()))

            # current_nurse = current_solution[random_room, random_shift]
            avail_nurses = list(nurses_per_shift[random_shift])

            avail_nurses.remove(current_nurse)
            new_nurse = random.sample(avail_nurses, k=1)[0]

            current_nurse_workload = workload_per_nurse_per_day[current_nurse][random_shift] - \
                workload_per_room_per_shift[random_room][random_shift]
            new_nurse_workload = workload_per_nurse_per_day[new_nurse][random_shift] + \
                workload_per_room_per_shift[random_room][random_shift]
            delta_excess_new = max(0, new_nurse_workload - instance.nurses[new_nurse].shifts[random_shift]) - max(
                0, workload_per_nurse_per_day[new_nurse][random_shift] - instance.nurses[new_nurse].shifts[random_shift])
            delta_excess_current = max(0, current_nurse_workload - instance.nurses[current_nurse].shifts[random_shift]) - max(
                0, workload_per_nurse_per_day[current_nurse][random_shift] - instance.nurses[current_nurse].shifts[random_shift])
            delta_excess = delta_excess_new + delta_excess_current


            current_nurse_skill = sum(max(0, i - instance.nurses[current_nurse].skill_level) for i in skills_per_room_per_shift[random_room][random_shift])
            new_nurse_skill = sum(max(0, i - instance.nurses[new_nurse].skill_level) for i in skills_per_room_per_shift[random_room][random_shift])
            delta_skill = new_nurse_skill - current_nurse_skill

            delta_coc = 0
            day = instance.shift_to_day(random_shift)
            for p in room_sol.room_day_guests[random_room, day]:
                counter_current_nurse = nurses_per_patient[p][current_nurse]
                if counter_current_nurse == 1:
                    delta_coc -= 1
                counter_new_nurse = nurses_per_patient[p][new_nurse]
                if counter_new_nurse == 0:
                    delta_coc += 1

            delta_total = instance.weights["nurse_eccessive_workload"] * delta_excess + \
                instance.weights["continuity_of_care"] * delta_coc + \
                instance.weights["room_nurse_skill"] * delta_skill
            
            iterations_without_improvement +=1
            if delta_total < 0 or random.random() < math.exp(-delta_total / temperature):
                
                current_solution[random_room, random_shift] = new_nurse
                workload_per_nurse_per_day[current_nurse][random_shift] = current_nurse_workload
                workload_per_nurse_per_day[new_nurse][random_shift] = new_nurse_workload
                excess_workload_per_nurse[new_nurse] += delta_excess_new
                excess_workload_per_nurse[current_nurse] += delta_excess_current
                skill_diff_per_nurse[current_nurse] -= current_nurse_skill
                skill_diff_per_nurse[new_nurse] += new_nurse_skill
                new_nurse_coc = 0
                for p in room_sol.room_day_guests[random_room, day]:
                    nurses_per_patient[p][new_nurse] += 1
                    nurses_per_patient[p][current_nurse] -= 1
                    coc_per_patient[p] = sum(1 for val in nurses_per_patient[p].values() if val != 0)
                    new_nurse_coc += 1 - (nurses_per_patient[g][new_nurse]/(instance.guests[p].length_of_stay*3))

                new_coc = instance.weights["continuity_of_care"] * sum(
                    coc_per_patient.values()) 
                new_skill = instance.weights["room_nurse_skill"] * sum(skill_diff_per_nurse.values())
                new_excess = instance.weights["nurse_eccessive_workload"] * sum(excess_workload_per_nurse.values()) 
                new_obj = new_coc + new_skill + new_excess

                new_cost = instance.weights["room_nurse_skill"] *new_nurse_skill
                new_cost += instance.weights["nurse_eccessive_workload"]*workload_per_room_per_shift[random_room][random_shift]
                new_cost += instance.weights["continuity_of_care"]  * new_nurse_coc 
                cost[current_nurse,random_room,random_shift] = 0 
                cost[new_nurse,random_room,random_shift] = new_cost

                current_objective = new_obj
                if current_objective < best_objective:
                    iterations_without_improvement = 0
                    improving_solutions += 1
                    best_objective = current_objective
                    best_solution = copy.deepcopy(current_solution)
                    
