import sys
import os
import time
import threading
import queue
import copy
import math
import random
import traceback
from gurobipy import *

def isFixedVar(var):
  return isinstance(var, int)

def isZeroVar(var):
  return isinstance(var, int) and var == 0

def evalVar(var):
  if isinstance(var, int):
    return float(var)
  if isinstance(var, float):
    return var
  else:
    return var.x

class FinalNursesTask:

  def __init__(self, sol, *arg, **kwargs):
    self._solution = sol
    self._best_objective_value = float('inf')
    super().__init__(*arg, **kwargs)

  def event_solution(self, instance, algo, room_shift_nurse):
    '''
    Returns the costs of this room/shift-nurse assignment and updates the incumbent.
    '''
    sol = self._solution

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

#    print(f'true objective_value = {objective_value} <? {self._best_objective_value}')

    if objective_value < self._best_objective_value:
      if not sol.has_nurses or objective_value < sol.costs_total:
        self._best_objective_value = objective_value
        self._best_room_shift_nurse = room_shift_nurse.copy()
        sol.set_room_shift_nurse(self._best_room_shift_nurse, algo.current_time())
        algo.event_new_final_solution(sol)

    return objective_value





  def solve_with_lazy_guest(self, instance, algo, room_sol):

    env = Env("", params = {
      'outputflag': 0,
      'timeLimit': max(0.0, algo.remaining_time_total()),
      'threads': 1,
      'cuts': 3,
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
    print(f'#room-shift-nurse vars: {len(model._var_room_shift_nurse)}')

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
  
    # Shift minimum costs (objective cut per shift).
    cons_shift_min_cost = {}
    if room_sol.has_shift_min_cost:
      for s in instance.shifts:
        lhs = quicksum( var.obj * var for key,var in model._var_room_shift_nurse.items() if not isFixedVar(var) and key[1] == s and var.obj != 0 ) + \
          quicksum( var.obj * var for key,var in model._var_nurse_shift_excess.items() if not isFixedVar(var) and key[1] == s and var.obj != 0 )
        cons_shift_min_cost[s] = model.addConstr( lhs >= room_sol.shift_min_cost[s], name=f'ShiftMinCosts_{s}')

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
#        for s in instance.shifts:
#          all_integer = True
#          foo = {}
#          for r in instance.rooms:
#            for n in instance.nurses:
#              val = val_room_shift_nurse.get((r,s,n), 0.0)
#              if val > 0.01:
#                foo[n,r] = val
#                if val < 0.99:
#                  all_integer = False
#          if not all_integer:
#            print(f'Shift {s}: {foo}')
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
          self.event_solution(instance, algo, room_shift_nurse)

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
    self.event_solution(instance, algo, room_shift_nurse)

    return model.objBound


  def solve_with_heuristic_best(self, thread_index, instance, algo, room_sol, only_local_search, init_nurse_sol, time_limit):

#    print(f'Starting SA for {instance}, {room_sol}, {only_local_search}, {init_nurse_sol}, {time_limit}')

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
              n = init_nurse_sol.get((r, s))
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

    runtime = min(time_limit, algo.remaining_time_total())
    print(f'FIN{thread_index}: Starting SA with time limit {runtime}. Total remaining time is {algo.remaining_time_total():.1f}s.')
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

        if  elapsed >= runtime - 1:
#            print(f"STOP SA after {iterations} iterations due to runtime limit {runtime}: Found {improving_solutions} improving solutions. {elapsed}s {iterations_without_improvement} iterations without improvement {algo.remaining_time_total()}")
            break


        if iterations_without_improvement >= 5000:
#            print(f"STOP SA after {iterations} iterations due to iterations without improvement {iterations_without_improvement}: Found {improving_solutions} improving solutions. {elapsed}s {iterations_without_improvement} iterations without improvement {algo.remaining_time_total()}")
            break

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

    self.event_solution(instance, algo, best_solution)

  def solve_with_guestwisecare(self, thread_index, instance, algo, room_sol, with_disaggregation):

    started = time.time()

#    print(f'\n\n======== solve_with_guestwisecare =======  {algo.remaining_time_total()} - {heuristic_time_limit}\n')

    env = Env("", params = {
      'outputflag': 0,
      'timeLimit': max(0.0, algo.remaining_time_total()),
      'threads': 1,
      'method': 2, 
#      'heuristics': 0.5 if len(instance.days) > 21 else 0.05,
      })
    model = Model("Nurses", env=env)

    model.addVar(lb=1.0, ub=1.0, obj=room_sol.costs_no_nurse)

    # Whether room/shift has this nurse.
    model._var_room_shift_nurse = {}
    for n,n_data in instance.nurses.items():
      for s in n_data.shifts:
        d = instance.shift_to_day(s)
        for r in instance.rooms:
          gs = room_sol.room_day_guests.get((r,d))
          if gs:
            sum_skill_deviation = 0
            for g in gs:
              g_data = instance.guests[g]
              rel_s = s - 3 * room_sol.guests_day[g]
              g_skill_level = g_data.skill_level_required[rel_s]
              sum_skill_deviation += max(0, g_skill_level - n_data.skill_level)
            model._var_room_shift_nurse[r,s,n] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['room_nurse_skill'] * sum_skill_deviation, name=f'RoomShiftNurse_{r}_{s}_{n}')
#    print(f'#room-shift-nurse vars: {len(model._var_room_shift_nurse)}')

    # Excess workload variables.
    model._var_nurse_shift_excess = {}
    for n,n_data in instance.nurses.items():
      for s in n_data.shifts:
        model._var_nurse_shift_excess[n,s] = model.addVar(vtype=GRB.CONTINUOUS, obj=instance.weights['nurse_eccessive_workload'], name=f'NurseShiftExcess_{n}_at_{s}')
#    print(f'#nurse-shift-excess vars: {len(model._var_nurse_shift_excess)}')

    # Whether a nurse serves a guest.
    model._var_nurse_guest = {}
    for n in instance.nurses:
      for g in room_sol.guests_room:
        model._var_nurse_guest[n,g] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['continuity_of_care'], name=f'NurseGuest_{n}_{g}')
#    print(f'#nurse-guest vars: {len(model._var_nurse_guest)}')


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
#    print(f'#room-has-nurse cons: {len(cons_room_has_nurse)}')

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
                cons_nurse_guest_disagg[n,g,s] = model.addConstr( var <= model._var_nurse_guest[n,g], f'disagg#{n}#{g}#{s}')
#    print(f'#disaggregated implications: {len(cons_nurse_guest_disagg)}')

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

      if algo.remaining_time_total() < 0.1:
        model.terminate()

      best_bound = None
      if where == GRB.Callback.MIPNODE:
        best_bound = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
      if where == GRB.Callback.MIPSOL:
        best_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
      if where == GRB.Callback.MIP:
        best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
      if best_bound is not None:
#        print(f'Checking bound {best_bound} vs. {algo.final_incumbent_cost}')
        if best_bound >= algo.final_incumbent_cost:
#          print(f'Aborting!')
          model.terminate()

      if where == GRB.Callback.MIPSOL:
        keys = [ key for key,var in model._var_room_shift_nurse.items() if not isFixedVar(var) ]
        vals = model.cbGetSolution( [ var for var in model._var_room_shift_nurse.values() if not isFixedVar(var) ] )
        room_shift_nurse = { key[:2]:key[2] for key,val in zip(keys, vals) if val > 0.5 }

#        old_best_mip = room_sol.costs_total
        self.event_solution(instance, algo, room_shift_nurse)
#        new_best_mip = room_sol.costs_total
#        if new_best_mip != old_best_mip:
#          print(f'Callback MIP: {old_best_mip} -> {new_best_mip}')

#        if heuristic_callback:  
#          old_best = room_sol.costs_total
#          self.solve_with_heuristic_best(thread_index, instance, algo, room_sol, heuristic_local, room_shift_nurse, heuristic_time_limit)
#          new_best = room_sol.costs_total
#          if new_best != old_best:
#            print(f'Callback SA: {old_best} -> {new_best}; MIP was {old_best_mip} -> {new_best_mip}')

    model.optimize( solver_callback )

    if not model.solCount:
      if model.status == GRB.INTERRUPTED:
        print(f'FIN{thread_index}: Interrupted before finding a feasible solution after {time.time() - started:.1f}s')
      else:
        print(f'FIN{thread_index}: Did not finding a feasible solution after {time.time() - started:.1f}s')
      return None

    room_shift_nurse = {}
    for r in instance.rooms:
      for s in instance.shifts:
        for n in instance.nurses:
          var = model._var_room_shift_nurse.get((r,s,n), None)
          if var is not None and var.x > 0.5:
            assert (r,s) not in room_shift_nurse
            room_shift_nurse[r,s] = n
    self.event_solution(instance, algo, room_shift_nurse)

    # Final solution also polished with SA.
#    if not heuristic_callback and not model.status == GRB.INTERRUPTED:
#      old_best = room_sol.costs_total
#      self.solve_with_heuristic_best(thread_index, instance, algo, room_sol, heuristic_local, room_shift_nurse, heuristic_time_limit)
#      new_best = room_sol.costs_total
#      if new_best != old_best:
#        print(f'{old_best} -> {new_best}')

    if model.status == GRB.INTERRUPTED:
      print(f'FIN{thread_index}: Interrupted with feasible solutions after {time.time() - started:.1f}s.')
    elif model.status == GRB.TIME_LIMIT:
      print(f'FIN{thread_index}: Time limit reached after {time.time() - started:.1f}s.')
    elif model.status == GRB.OPTIMAL:
      print(f'FIN{thread_index}: Solved to optimality after {time.time() - started:.1f}s.')
    else:
      print(f'FIN{thread_index}: Unknown status after {time.time() - started:.1f}s.')
  
  
  
  def solve_with_restricted_guestwisecare(self, instance, algo, room_sol, with_disaggregation):

    env = Env("", params = {
      'outputflag': 1,
      'timeLimit': max(0, algo.remaining_time_total()),
      'threads': 1,
      })
    model = Model("Nurses", env=env)

    prev_guest_nurses = { g: set() for g in room_sol.guests_room }
    if room_sol.has_nurses:
      for r in instance.rooms:
        for s in instance.shifts:
          for n in instance.nurses:
            if room_sol.room_shift_nurse.get((r,s), None) == n:
              gs = room_sol.room_day_guests.get((r,instance.shift_to_day(s)), [])
              for g in gs:
                prev_guest_nurses[g].add(n)
    for g in prev_guest_nurses:
      print(f'Previous sol: guest {g} is served by {len(prev_guest_nurses[g])} / {len(instance.nurses)} nurses.')

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
    print(f'#room-shift-nurse vars: {len(model._var_room_shift_nurse)}')

    # Whether a nurse serves a guest.
    model._var_nurse_guest = {}
    for n in instance.nurses:
      for g in room_sol.guests_room:
        if n in prev_guest_nurses[g]:
          model._var_nurse_guest[n,g] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['continuity_of_care'], name=f'NurseGuest_{n}_{g}')
    print(f'#nurse-guest vars: {len(model._var_nurse_guest)}')

    # Excess workload variables.
    model._var_nurse_shift_excess = {}
    for n,n_data in instance.nurses.items():
      for s in n_data.shifts:
        model._var_nurse_shift_excess[n,s] = model.addVar(vtype=GRB.CONTINUOUS, obj=instance.weights['nurse_eccessive_workload'], name=f'NurseShiftExcess_{n}_at_{s}')
    print(f'#nurse-shift-excess vars: {len(model._var_nurse_shift_excess)}')

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

    # Nurse in room implies that nurse cares about patients.
    cons_nurse_guest = {}
    for n in instance.nurses:
      for g,r in room_sol.guests_room.items():
        if (n,g) not in model._var_nurse_guest:
          continue
        lhs = []
        g_data = instance.guests[g]
        d_first = room_sol.guests_day[g]
        d_last = min(instance.last_day, d_first + g_data.length_of_stay - 1)
        for s in range(3*d_first, 3*(d_last + 1)):
          if (r,s,n) in model._var_room_shift_nurse and not isinstance(model._var_room_shift_nurse[r,s,n], int):
            lhs.append( model._var_room_shift_nurse[r,s,n] )
        cons_nurse_guest[n,g] = model.addConstr( quicksum( lhs ) <= len(lhs) * model._var_nurse_guest[n,g] )
    print(f'#aggregated implications: {len(cons_nurse_guest)}')

    # Disaggregation: nurse does not care about guest g => nurse is not there in any shift.
    cons_nurse_guest_disagg = {}
    if with_disaggregation:
      for n,n_data in instance.nurses.items():
        for g,r in room_sol.guests_room.items():
          if (n,g) not in model._var_nurse_guest:
            continue
          for s in n_data.shifts:
            if instance.shift_to_day(s) in room_sol.guest_stay(g):
              var = model._var_room_shift_nurse.get((r,s,n), None)
              if var is not None and not isinstance(var, int):
                cons_nurse_guest_disagg[n,g,s] = model.addConstr( var <= model._var_nurse_guest[n,g], f'disagg#n#g#s')
    print(f'#disaggregated implications: {len(cons_nurse_guest_disagg)}')

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
        cons_guest_shift_nurse_cover[g,s] = model.addConstr( quicksum( model._var_nurse_guest.get((n,g), 0.0) for n in possible_nurses ) >= 1 )
    print(f'#guest-shift-nurse-cover: {len(cons_guest_shift_nurse_cover)}')

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
    print(f'#workload: {len(cons_workload)}')

    # Shift minimum costs (objective cut per shift).
    cons_shift_min_cost = {}
    if room_sol.has_shift_min_cost:
      for s in instance.shifts:
        lhs = quicksum( var.obj * var for key,var in model._var_room_shift_nurse.items() if not isFixedVar(var) and key[1] == s and var.obj != 0 ) + \
          quicksum( var.obj * var for key,var in model._var_nurse_shift_excess.items() if not isFixedVar(var) and key[1] == s and var.obj != 0 )
        cons_shift_min_cost[s] = model.addConstr( lhs >= room_sol.shift_min_cost[s], name=f'ShiftMinCosts_{s}')
    print(f'#shift-min-cost objective cuts: {len(cons_shift_min_cost)}')

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
        self.event_solution(instance, algo, room_shift_nurse)

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
    self.event_solution(instance, algo, room_shift_nurse)


  def solve_with_roomwisecare(self, instance, algo, room_sol, with_disaggregation):

    env = Env("", params = {
      'outputflag': 1,
      'timeLimit': max(0.0, algo.remaining_time_total()),
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
        room_factor = len(room_guests[r])
#        room_factor = 1 #instance.rooms[r].capacity
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
        self.event_solution(instance, algo, room_shift_nurse)

    if room_sol.has_nurses:
      for r in instance.rooms:
        for s in instance.shifts:
          for n in instance.nurses:
            var = model._var_room_shift_nurse.get((r,s,n), 0)
            if not isFixedVar(var):
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
    self.event_solution(instance, algo, room_shift_nurse)

    return model.objBound






  def benders_nurse_guest_main(self, instance, algo, room_sol):

    print(f'Considering solution:')
    for g,d in room_sol.guests_day.items():
      r = room_sol.guests_room[g]
      last = max(room_sol.guest_stay(g))
      print(f'On day {d}, guest {g} is admitted to room {r} for stay [{d},{last}]')

    env = Env("", params = {
      'outputflag': 1,
      'timeLimit': max(0.0, algo.remaining_time_total()),
      'threads': 1,
      })
    model = Model("Nurses", env=env)

    # Whether room/shift has this nurse.
#    model._var_room_shift_nurse = {}
#    for n,n_data in instance.nurses.items():
#      for s in n_data.shifts:
#        d = instance.shift_to_day(s)
#        for r in instance.rooms:
#          if (r,d) in room_sol.room_day_guests:
#            sum_skill_deviation = 0
#            for g in room_sol.room_day_guests[r,d]:
#              g_data = instance.guests[g]
#              rel_s = s - 3 * room_sol.guests_day[g]
#              g_skill_level = g_data.skill_level_required[rel_s]
#              sum_skill_deviation += max(0, g_skill_level - n_data.skill_level)
#            model._var_room_shift_nurse[r,s,n] = model.addVar(ub=1.0, obj=instance.weights['room_nurse_skill'] * sum_skill_deviation, name=f'RoomShiftNurse_{r}_{s}_{n}')
#          else:
#            model._var_room_shift_nurse[r,s,n] = 0
#    print(f'#room-shift-nurse: {len(model._var_room_shift_nurse)}')

    # Whether a nurse serves a guest.
    model._var_nurse_guest = {}
    for n in instance.nurses:
      for g in room_sol.guests_room:
        model._var_nurse_guest[n,g] = model.addVar(vtype=GRB.CONTINUOUS, obj=instance.weights['continuity_of_care'], name=f'NurseGuest_{n}_{g}')
    model.addVar(vtype=GRB.BINARY)
#    print(f'#nurse-guest vars: {len(model._var_nurse_guest)}')

    # Excess workload counts per shift.
    model._var_shift_costs = {}
    for s in instance.shifts:
        model._var_shift_costs[s] = model.addVar(obj=1.0, name=f'ShiftCosts_{s}')

    model.update()

    # Each room needs a nurse.
#    cons_room_has_nurse = {}
#    for r in instance.rooms:
#      for s in instance.shifts:
#        d = instance.shift_to_day(s)
#        gs = room_sol.room_day_guests.get((r,d), [])
#        if gs:
#          cons_room_has_nurse[r,s] = model.addConstr(
#            quicksum( model._var_room_shift_nurse[r,s,n] for n,n_data in instance.nurses.items() if s in n_data.shifts ) == 1,
#            name=f'RoomHasNurse_{r}_at_{s}')
#    print(f'#room-has-nurse: {len(cons_room_has_nurse)}')

    # Nurse in room implies that nurse cares about patients.
#    cons_nurse_guest = {}
#    for n in instance.nurses:
#      for g,r in room_sol.guests_room.items():
#        lhs = []
#        g_data = instance.guests[g]
#        d_first = room_sol.guests_day[g]
#        d_last = min(instance.last_day, d_first + g_data.length_of_stay - 1)
#        for s in range(3*d_first, 3*(d_last + 1)):
#          if (r,s,n) in model._var_room_shift_nurse and not isinstance(model._var_room_shift_nurse[r,s,n], int):
#            lhs.append( model._var_room_shift_nurse[r,s,n] )
#        cons_nurse_guest[n,g] = model.addConstr( quicksum( lhs ) <= len(lhs) * model._var_nurse_guest[n,g] )
#    print(f'#aggregated implications: {len(cons_nurse_guest)}')

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
#    cons_workload = {}
#    for n,n_data in instance.nurses.items():
#      for s,load in n_data.shifts.items():
#        room_workload = { r: 0 for r in instance.rooms }
#        d = instance.shift_to_day(s)
#        for r in instance.rooms:
#          for g in room_sol.room_day_guests.get((r,d), []):
#            workload_produced = instance.guests[g].workload_produced[ s - 3 * room_sol.guests_day[g] ]
#            room_workload[r] += workload_produced
#
#          cons_workload[n,s] = model.addConstr(
#            quicksum( room_workload[r] * model._var_room_shift_nurse.get((r,s,n), 0.0) for r in instance.rooms )
#              <= load + model._var_nurse_shift_excess[n,s],
#            name=f'NurseCapacity_{n}_{s}')
#    print(f'#workload: {len(cons_workload)}')

    # Shift minimum costs (objective cut per shift).
#    cons_shift_min_cost = {}
#    if room_sol.has_shift_min_cost:
#      for s in instance.shifts:
#        lhs = quicksum( var.obj * var for key,var in model._var_room_shift_nurse.items() if not isFixedVar(var) and key[1] == s and var.obj != 0 ) + \
#          quicksum( var.obj * var for key,var in model._var_nurse_shift_excess.items() if not isFixedVar(var) and key[1] == s and var.obj != 0 )
#        cons_shift_min_cost[s] = model.addConstr( lhs >= room_sol.shift_min_cost[s], name=f'ShiftMinCosts_{s}')

#    if room_sol.has_nurses:
#      for r in instance.rooms:
#        for s in instance.shifts:
#          for n in instance.nurses:
#            var = model._var_room_shift_nurse.get((r,s,n), None)
#            if var is not None:
#              var.start = 1 if room_sol.room_shift_nurse.get((r,s), None) == n else 0
    
    sub_env = Env("", params = {
      'outputflag': 1,
      'threads': 1,
      'heuristics': 0, # tuned
      'cuts': 2, # tuned 
      })

    sub_model = {}

    # Prepare subproblems
    for s in instance.shifts:
      d = instance.shift_to_day(s)

      sub_model[s] = Model(f'NurseShift_{s}', env=sub_env)

      nurses = [ n for n,n_data in instance.nurses.items() if s in n_data.shifts ]
#      print(f'Shift-based model for {len(nurses)} nurses in shift {s}.')

      # Room/nurse variables.
      sub_model[s]._var_room_nurse = {}
      sub_model[s]._obj_room_nurse = {}
      for n in nurses:
        n_data = instance.nurses[n]
        for r in instance.rooms:
          gs = room_sol.room_day_guests.get((r,d), [])
          if gs:
            # Sum of skill level deviations
            sum_skill_deviation = 0
            # Number of guests this nurse would be new to.
            for g in room_sol.room_day_guests[r,d]:
              g_data = instance.guests[g]
              rel_s = s - 3 * room_sol.guests_day[g]
              g_skill_level = g_data.skill_level_required[rel_s]
              sum_skill_deviation += max(0, g_skill_level - n_data.skill_level)
            sub_model[s]._obj_room_nurse[r,n] = instance.weights['room_nurse_skill'] * sum_skill_deviation
            sub_model[s]._var_room_nurse[r,n] = sub_model[s].addVar(name=f'RoomNurse_{r}_{n}')

      # Excess workload variables.
      sub_model[s]._var_nurse_excess = {}
      for n in nurses:
        sub_model[s]._var_nurse_excess[n] = sub_model[s].addVar(vtype=GRB.CONTINUOUS, obj=instance.weights['nurse_eccessive_workload'], name=f'NurseShiftExcess_{n}')

      sub_model[s].update()

      # Each room needs a nurse.
      sub_model[s]._cons_room_has_nurse = {}
      for r in instance.rooms:
        gs = room_sol.room_day_guests.get((r,d), [])
        if gs:
          sub_model[s]._cons_room_has_nurse[r] = sub_model[s].addConstr(
            quicksum( sub_model[s]._var_room_nurse[r,n] for n,n_data in instance.nurses.items() if s in n_data.shifts ) == 1, name=f'RoomHasNurse_{r}')

      # Nurse workload
      sub_model[s]._cons_workload = {}
      for n in nurses:
        n_data = instance.nurses[n]
        load = n_data.shifts.get(s, 0)
        if load:
          room_workload = { r: 0 for r in instance.rooms }
          for r in instance.rooms:
            for g in room_sol.room_day_guests.get((r,d), []):
              workload_produced = instance.guests[g].workload_produced[ s - 3 * room_sol.guests_day[g] ]
              room_workload[r] += workload_produced
          sub_model[s]._cons_workload[n] = sub_model[s].addConstr(
            quicksum( room_workload[r] * sub_model[s]._var_room_nurse.get((r,n), 0.0) for r in instance.rooms )
              <= load + sub_model[s]._var_nurse_excess[n],
            name=f'NurseCapacity_{n}')

    def solver_callback(model, where):

      if where == GRB.Callback.MIPSOL or where == GRB.Callback.MIPNODE:
        keys = [ key for key,var in model._var_nurse_guest.items() if not isFixedVar(var) ]
        vals = model.cbGetSolution( [ var for var in model._var_nurse_guest.values() if not isFixedVar(var) ] )
        nurse_guest_vals = { key:val for key,val in zip(keys, vals) if val > 0.01 }
      elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
        keys = [ key for key,var in model._var_nurse_guest.items() if not isFixedVar(var) ]
        vals = model.cbGetNodeRel( [ var for var in model._var_nurse_guest.values() if not isFixedVar(var) ] )
        val_guest_nurses = { key:val for key,val in zip(keys, vals) if val > 0.01 }
      else:
        return

      # Feasibility cuts.
      upper_bounds = {}
      num_feasibility_cuts = 0
      for key,gs in room_sol.room_day_guests.items():
        if not gs:
          continue
        r,d = key
        for s in range(len(instance.shift_types)*d, len(instance.shift_types)*(d+1)):
          upper_bounds[s,r] = {}
          sum_bounds = 0.0
          nurses = [ n for n,n_data in instance.nurses.items() if s in n_data.shifts ]
          for n in nurses:
            min_value = float('inf')
            min_guest = None
            for g in gs:
              val = nurse_guest_vals.get((n,g), 0.0)
              if val < min_value:
                min_guest,min_value = g,val
            upper_bounds[s,r][n] = (min_value,min_guest)
            sum_bounds += min_value
          if sum_bounds < 0.999:
#            print(f'Current sol is infeasible for shift {s} (day {d}) and room {r}. Sum of upper bounds is {sum_bounds}.')
            model.cbLazy( quicksum( model._var_nurse_guest[n,upper_bounds[s,r][n][1]] for n in nurses ) >= 1 )
            num_feasibility_cuts += 1
      if num_feasibility_cuts:
        print(f'Generated {num_feasibility_cuts} feasibility cuts.')
        return

      print(f'LP-feasible.')

      sys.exit(41)

      for g in room_sol.guests_day:
        print(f'{g} cared by {[ key[0] for key in nurse_guest_vals if key[1] == g]}')

      for s in instance.shifts:
        d = instance.shift_to_day(s)
        for r in instance.rooms:
          gs = room_sol.room_day_guests[r,d]
          for n,n_data in instance.nurses.items():
            if s not in n_data.shifts:
              continue
            if (r,n) not in sub_model[s]._var_room_nurse:
              continue
            nurse_allowed = True
            for g in gs:
              if (n,g) not in nurse_guest_vals:
                nurse_allowed = False
                print(f'Shift {s} on day {d}: nurse {n} disallowed for room {r} by guest {g}.')
                break
            if nurse_allowed:
              sub_model[s]._var_room_nurse[r,n].obj = sub_model[s]._obj_room_nurse[r,n]
              print(f'Shift {s} on day {d}: nurse {n} allowed for room {r}.')
            else:
              sub_model[s]._var_room_nurse[r,n].obj = 1.0e6
        sub_model[s].optimize()
        for key,var in sub_model[s]._var_room_nurse.items():
          if var.x > 0.01:
            print(f'Nurse {key[1]} in room {key[0]}: {var.x}')

#          for n,cons in sub_model[s]._cons_workload.items():
#            print(f'Nurse {n}\'s workload constraint has dual {cons.pi}')
            

#        model.cbLazy(  )

#    model.write('prob-final.lp')
#    model.params.threads = 1
    model.params.lazyConstraints = 1
#    model.params.heuristics = 0.01
#    model.params.cuts = 0
    model.optimize( solver_callback )

    if not model.solCount:
      return None

#    room_shift_nurse = {}
#    for r in instance.rooms:
#      for s in instance.shifts:
#        for n in instance.nurses:
#          var = model._var_room_shift_nurse.get((r,s,n), None)
#          if var is not None and var.x > 0.5:
#            assert (r,s) not in room_shift_nurse
#            room_shift_nurse[r,s] = n
#    self.event_solution(instance, algo, room_shift_nurse)





  
  
  
  def solve_with_restricted_guestwisecare(self, instance, algo, room_sol, with_disaggregation):

    env = Env("", params = {
      'outputflag': 1,
      'timeLimit': max(0, algo.remaining_time_total()),
      'threads': 1,
      })
    model = Model("Nurses", env=env)

    prev_guest_nurses = { g: set() for g in room_sol.guests_room }
    if room_sol.has_nurses:
      for r in instance.rooms:
        for s in instance.shifts:
          for n in instance.nurses:
            if room_sol.room_shift_nurse.get((r,s), None) == n:
              gs = room_sol.room_day_guests.get((r,instance.shift_to_day(s)), [])
              for g in gs:
                prev_guest_nurses[g].add(n)
    for g in prev_guest_nurses:
      print(f'Previous sol: guest {g} is served by {len(prev_guest_nurses[g])} / {len(instance.nurses)} nurses.')

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

    # Whether a nurse serves a guest.
    model._var_nurse_guest = {}
    for n in instance.nurses:
      for g in room_sol.guests_room:
        if n in prev_guest_nurses[g]:
          model._var_nurse_guest[n,g] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['continuity_of_care'], name=f'NurseGuest_{n}_{g}')
    print(f'#nurse-guest vars: {len(model._var_nurse_guest)}')

    # Excess workload variables.
    model._var_nurse_shift_excess = {}
    for n,n_data in instance.nurses.items():
      for s in n_data.shifts:
        model._var_nurse_shift_excess[n,s] = model.addVar(vtype=GRB.CONTINUOUS, obj=instance.weights['nurse_eccessive_workload'], name=f'NurseShiftExcess_{n}_at_{s}')
    print(f'#nurse-shift-excess vars: {len(model._var_nurse_shift_excess)}')

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

    # Nurse in room implies that nurse cares about patients.
    cons_nurse_guest = {}
    for n in instance.nurses:
      for g,r in room_sol.guests_room.items():
        if (n,g) not in model._var_nurse_guest:
          continue
        lhs = []
        g_data = instance.guests[g]
        d_first = room_sol.guests_day[g]
        d_last = min(instance.last_day, d_first + g_data.length_of_stay - 1)
        for s in range(3*d_first, 3*(d_last + 1)):
          if (r,s,n) in model._var_room_shift_nurse and not isinstance(model._var_room_shift_nurse[r,s,n], int):
            lhs.append( model._var_room_shift_nurse[r,s,n] )
        cons_nurse_guest[n,g] = model.addConstr( quicksum( lhs ) <= len(lhs) * model._var_nurse_guest[n,g] )
    print(f'#aggregated implications: {len(cons_nurse_guest)}')

    # Disaggregation: nurse does not care about guest g => nurse is not there in any shift.
    cons_nurse_guest_disagg = {}
    if with_disaggregation:
      for n,n_data in instance.nurses.items():
        for g,r in room_sol.guests_room.items():
          if (n,g) not in model._var_nurse_guest:
            continue
          for s in n_data.shifts:
            if instance.shift_to_day(s) in room_sol.guest_stay(g):
              var = model._var_room_shift_nurse.get((r,s,n), None)
              if var is not None and not isinstance(var, int):
                cons_nurse_guest_disagg[n,g,s] = model.addConstr( var <= model._var_nurse_guest[n,g], f'disagg#n#g#s')
    print(f'#disaggregated implications: {len(cons_nurse_guest_disagg)}')

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
        cons_guest_shift_nurse_cover[g,s] = model.addConstr( quicksum( model._var_nurse_guest.get((n,g), 0.0) for n in possible_nurses ) >= 1 )
    print(f'#guest-shift-nurse-cover: {len(cons_guest_shift_nurse_cover)}')

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
    print(f'#workload: {len(cons_workload)}')

    # Shift minimum costs (objective cut per shift).
    cons_shift_min_cost = {}
    if room_sol.has_shift_min_cost:
      for s in instance.shifts:
        lhs = quicksum( var.obj * var for key,var in model._var_room_shift_nurse.items() if not isFixedVar(var) and key[1] == s and var.obj != 0 ) + \
          quicksum( var.obj * var for key,var in model._var_nurse_shift_excess.items() if not isFixedVar(var) and key[1] == s and var.obj != 0 )
        cons_shift_min_cost[s] = model.addConstr( lhs >= room_sol.shift_min_cost[s], name=f'ShiftMinCosts_{s}')
    print(f'#shift-min-cost objective cuts: {len(cons_shift_min_cost)}')

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
        self.event_solution(instance, algo, room_shift_nurse)

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
    self.event_solution(instance, algo, room_shift_nurse)

############################################

  def solve_rolling_horizon_flexible(self, instance, algo, room_sol, with_disaggregation, guests_nurses, days, time_limit):
    '''
    Solve MIP, with nurse/room/guest combinations only for given days.
    Guests without guests_nurses entry don't have continuity-of-care variables.
    '''

    days_shifts = []
    for d in days:
      for s in range(len(instance.shift_types)*d, len(instance.shift_types)*(d+1)):
        days_shifts.append((d,s))
#    print(days_shifts)

    env = Env("", params = {
      'outputflag': 1,
      'timeLimit': min(time_limit, max(0.0, algo.remaining_time_total())),
      'threads': 1,
      })
    model = Model("Nurses", env=env)

    # Whether room/shift has this nurse.
    model._var_room_shift_nurse = {}
    for n,n_data in instance.nurses.items():
      for d,s in days_shifts:
        if s in n_data.shifts:
          for r in instance.rooms:
            gs = room_sol.room_day_guests.get((r,d))
            if gs:
              sum_skill_deviation = sum( max(0, instance.guests[g].skill_level_required[s - 3 * room_sol.guests_day[g]] - n_data.skill_level) for g in gs )
              model._var_room_shift_nurse[r,s,n] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['room_nurse_skill'] * sum_skill_deviation, name=f'RoomShiftNurse_{r}_{s}_{n}')
#    print(f'#room-shift-nurse vars: {len(model._var_room_shift_nurse)}')

    # Excess workload variables.
    model._var_nurse_shift_excess = {}
    for n,n_data in instance.nurses.items():
      for d,s in days_shifts:
        if s in n_data.shifts:
          model._var_nurse_shift_excess[n,s] = model.addVar(vtype=GRB.CONTINUOUS, obj=instance.weights['nurse_eccessive_workload'], name=f'NurseShiftExcess_{n}_at_{s}')
#    print(f'#excess-workload vars: {len(model._var_nurse_shift_excess)}')

    # Whether a nurse serves a guest.
    model._var_nurse_guest = {}
    for g,ns in guests_nurses.items():
      for n in ns:
        model._var_nurse_guest[n,g] = model.addVar(vtype=GRB.BINARY, obj=instance.weights['continuity_of_care'], name=f'NurseGuest_{n}_{g}')
#    print(f'#nurse-guest vars: {len(model._var_nurse_guest)}')

    model.update()

    # Each room needs a nurse.
    cons_room_has_nurse = {}
    for r in instance.rooms:
      for d,s in days_shifts:
        gs = room_sol.room_day_guests.get((r,d))
        if gs:
          cons_room_has_nurse[r,s] = model.addConstr(
            quicksum( model._var_room_shift_nurse[r,s,n] for n,n_data in instance.nurses.items() if s in n_data.shifts ) == 1,
            name=f'RoomHasNurse_{r}_at_{s}')
#    print(f'#room-has-nurse: {len(cons_room_has_nurse)}')

    # Nurse workload
    cons_workload = {}
    for n,n_data in instance.nurses.items():
      for s,load in n_data.shifts.items():
        d = instance.shift_to_day(s)
        if d not in days:
          continue
        room_workload = { r: 0 for r in instance.rooms }
        for r in instance.rooms:
          for g in room_sol.room_day_guests.get((r,d), []):
            room_workload[r] += instance.guests[g].workload_produced[ s - 3 * room_sol.guests_day[g] ]
        cons_workload[n,s] = model.addConstr(
          quicksum( room_workload[r] * model._var_room_shift_nurse.get((r,s,n), 0.0) for r in instance.rooms )
            <= load + model._var_nurse_shift_excess[n,s],
          name=f'NurseCapacity_{n}_{s}')
#    print(f'#workload: {len(cons_workload)}')

    # Nurse in room implies that nurse cares about patients.
    cons_nurse_guest = {}
    for n in instance.nurses:
      for g,r in room_sol.guests_room.items():
        if g not in guests_nurses:
          continue
        var = model._var_nurse_guest.get((n,g), 0) # We need rhs = 0 for the missing ones!
        lhs = []
        g_data = instance.guests[g]
        d_first = room_sol.guests_day[g]
        d_last = min(instance.last_day, d_first + g_data.length_of_stay - 1)
        for s in range(3*d_first, 3*(d_last + 1)):
          if (r,s,n) in model._var_room_shift_nurse and not isinstance(model._var_room_shift_nurse[r,s,n], int):
            lhs.append( model._var_room_shift_nurse[r,s,n] )
        cons_nurse_guest[n,g] = model.addConstr( quicksum( lhs ) <= len(lhs) * var, name=f'Agg#{n}#{g}')
#    print(f'#aggregated implications: {len(cons_nurse_guest)}')
    
    # Disaggregation: nurse does not care about guest g => nurse is not there in any shift.
    cons_nurse_guest_disagg = {}
    if with_disaggregation:
      for key,var_rsn in model._var_room_shift_nurse.items():
        r,s,n = key
        d = instance.shift_to_day(s)
        for g in room_sol.room_day_guests[r,d]:
          var_ng = model._var_nurse_guest.get((n,g))
          if var_ng is not None:
            cons_nurse_guest_disagg[n,g,s] = model.addConstr( var_rsn <= var_ng, f'disagg#{n}#{g}#{s}')
#    print(f'#disaggregated implications: {len(cons_nurse_guest_disagg)}')

#    if not os.path.exists('prob-test.lp'):
#    model.write(f'prob-test.lp')

    model.optimize()

    if model.solCount:
      result_assignment = {}
      for r,s,n in model._var_room_shift_nurse:
        val_rsn = model._var_room_shift_nurse[r,s,n].x
        if val_rsn > 0.5:
          result_assignment[r,s] = n
#          print(f'Room {r} with guests {set(room_sol.room_day_guests[r,instance.shift_to_day(s)])} in shift {s} served by nurse {n}.')

      result_guest_nurses = { g:[] for g in guests_nurses }
      for n,g in model._var_nurse_guest:
        val_ng = model._var_nurse_guest[n,g].x
        if val_ng > 0.5:
          result_guest_nurses[g].append(n)
#          print(f'Nurse {n} cares about {g} (active in [{room_sol.guests_day[g]},{room_sol.guest_stay(g)[-1]}].')

      return result_assignment,result_guest_nurses,model.mipgap
    else:
      return None,None,None


  def rolling_horizon_flexible(self, instance, algo, room_sol, with_disaggregation, max_num_guests, time_limit_per_run):

#    print(f'Considering solution with {len(room_sol.guests_day)} guests:')
    for g,d in room_sol.guests_day.items():
      r = room_sol.guests_room[g]
      last = max(room_sol.guest_stay(g))

    guests = list(room_sol.guests_day.keys())
    ordered_guests = sorted(room_sol.guests_day.keys(), key=lambda g: (room_sol.guest_stay(g)[-1], -room_sol.guests_day[g]))
    for i,g in enumerate(ordered_guests):
      d = room_sol.guests_day[g]
      r = room_sol.guests_room[g]
      last = max(room_sol.guest_stay(g))
#      print(f'On day {d}, guest {g} is admitted to room {r} for stay [{d},{last}]')

    first_day = 0
    fixed_guest_nurses = {}

    while True:
      guests_nurses = {}
      beyond_day = first_day
      while beyond_day < len(instance.days):
        beyond_day_guests = [ g for g in guests if room_sol.guests_day[g] == beyond_day if not g in fixed_guest_nurses ]
        if len(guests_nurses) + len(beyond_day_guests) <= max_num_guests:
          guests_nurses.update( { g:list(instance.nurses) for g in beyond_day_guests } )
          beyond_day += 1
        else:
          break

      if not guests_nurses:
        break

#      print(f'day range = [{first_day}, {beyond_day})')

      min_day = float('inf')
      min_guest = None
      max_day = 0
      for g in guests_nurses:
        if room_sol.guests_day[g] < min_day:
          min_guest = g
        min_day = min(min_day, room_sol.guests_day[g])
        max_day = max(max_day, room_sol.guest_stay(g)[-1])

#      print(f'min_day guest = {min_guest}')

      first_day = min_day  
#      print(f'updated day range = [{first_day}, {beyond_day}), maximum day is {max_day}')

#      print(f'#Considered guests = {len(guests_nurses)}')

      guests_nurses.update(fixed_guest_nurses)

      total_num_guest_nurses = sum( len(ns) for ns in guests_nurses.values() )

      print(f'FIN{thread_index}: {room_sol.label}: partial on [0,{max_day}], fixing on [{first_day},{beyond_day-1}] with {len(fixed_guest_nurses)} fixed and {len(guests_nurses)-len(fixed_guest_nurses)} variable guests, {total_num_guest_nurses} guest/nurse pairs.', flush=True)

      time_start = algo.current_time()

      _,result_fixed_guest_nurses,mip_gap = self.solve_rolling_horizon_flexible(instance, algo, room_sol, with_disaggregation, guests_nurses, range(0, max_day+1), time_limit_per_run)

      if result_fixed_guest_nurses is not None:
        num_new = 0
        for g,ns in result_fixed_guest_nurses.items():
          if room_sol.guest_stay(g)[-1] < beyond_day:
            fixed_guest_nurses[g] = list(ns)
          num_new += 1

        print(f'FINAL: {room_sol.label}: additionally fixing {num_new} guests\' nurses after {algo.current_time() - time_start:.1f}s with gap {100*mip_gap:.1f}%.', flush=True)
      else:
        print(f'FINAL: {room_sol.label}: failed; {algo.current_time() - time_start:.1f}s', flush=True)
        break
 
    if algo.remaining_time_total() > 1:
      total_num_guest_nurses = sum( len(ns) for ns in fixed_guest_nurses.values() )
      print(f'FINAL: {room_sol.label}: final on [0,{max(instance.days)}] with {len(fixed_guest_nurses)} fixed guests, {total_num_guest_nurses} guest/nurse pairs.', flush=True)

      time_start = algo.current_time()

      room_shift_nurse,_,mip_gap = self.solve_rolling_horizon_flexible(instance, algo, room_sol, with_disaggregation, fixed_guest_nurses, instance.days, 2 * time_limit_per_run)

      print(f'FINAL: {room_sol.label}: completed after {algo.current_time() - time_start:.1f}s with gap {100*mip_gap:.1f}%.', flush=True)

      self.event_solution(instance, algo, room_shift_nurse)

  def run(self, thread_index, instance, algo):

    print(f'FIN{thread_index}: Solving for {self._solution}. Remaining time: {algo.remaining_time_total():.1f}s.', flush=True)

#    self.solve_with_roomwisecare(instance, algo, self._solution, True)
#    self.solve_with_restricted_guestwisecare(instance, algo, self._solution, True)
#    self.solve_with_lazy_guest(instance, algo, self._solution)
#    self.solve_with_guestwisecare(instance, algo, self._solution, True)

#    self.rolling_horizon_flexible(instance, algo, self._solution, False, 100, time_limit_per_run)

    self.solve_with_guestwisecare(thread_index, instance, algo, self._solution, False)

class FinalNursesThread(threading.Thread):

  def __init__(self, thread_index, queue, instance, algo, *arg, **kwargs):
    self._thread_index = thread_index
    self._queue = queue
    self._instance = instance
    self._algo = algo
    super().__init__(*arg, **kwargs)

  def run(self):
    while True:
      try:
        task = self._queue.get(timeout=0.01)
      except queue.Empty:
        return
      if self._algo.remaining_time_total() > 0.5:
        try:
          task.run(self._thread_index, self._instance, self._algo)
        except Exception as e:
          print(f'Exception: {e}')
          traceback.print_stack()
          sys.exit(1)
      self._queue.task_done()


