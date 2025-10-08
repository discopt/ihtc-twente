import random
import sys
import threading
import time
from data import *
from theaters import *
from ortools.sat.python import cp_model

class RoomsThread(threading.Thread):

  def __init__(self, instance, algo, max_num_solutions, time_limit, *arg, **kwargs):
    self._instance = instance
    self._algo = algo
    self._time_limit = time_limit
    self._waiting_for_admissions = True
    self._max_num_solutions = max_num_solutions
    assert max_num_solutions > 0
    super().__init__(*arg, **kwargs)

  def signalNoMoreAdmissionSolutions(self):
    self._waiting_for_admissions = False

  def run(self):
    instance = self._instance
    algo = self._algo

    # Solution callback class 
    class ObjectiveSolutionPrinter(cp_model.CpSolverSolutionCallback):

      def __init__(self, instance, algo, guests_room_list, var_guest_room):
        self._instance = instance
        self._algo = algo
        self._time_started = algo.current_time()
        self._guests_room_list = guests_room_list
        self._var_guest_room = var_guest_room
        cp_model.CpSolverSolutionCallback.__init__(self)

      def on_solution_callback(self):
        obj = self.ObjectiveValue()
        guests_room = { key[0]:key[1] for key,var in self._var_guest_room.items() if self.value(var) == 1 }
        self._guests_room_list.append( (obj, guests_room, self._algo.current_time() - self._time_started) )


    first_sleep = True
    while algo.remaining_time_parallel() > 1:
      adm_sol = algo.process_best_admission_solution()
      if adm_sol is None:
        if self._waiting_for_admissions:
          if first_sleep:
            first_sleep = False
            print(f'ROOM: Sleeping.', flush=True)
          time.sleep(0.1)
          continue
        else:
          print(f'ROOM: Stopping.', flush=True)
          return

      first_sleep = True
      print(f'ROOM: CP for {adm_sol} out of {algo.num_unprocessed_admission_solutions + 1}.', flush=True)

      model = cp_model.CpModel()

      var_guest_room = {}
      for g in adm_sol.guests_day:
        g_data = instance.guests[g]
        if g_data.is_occupant:
          for r in instance.rooms:
            var_guest_room[g,r] = 0
          var_guest_room[g,g_data.room] = 1        
        else:
          for r in instance.rooms:
            if r in g_data.incompatible_rooms:
              var_guest_room[g,r] = 0
            else:
              var_guest_room[g,r] = model.new_bool_var(f'PatRoom_{g}_in_{r}')

      var_room_day_genderA = {}
      for r in instance.rooms:
        for d in instance.days:
          var_room_day_genderA[r,d] = model.new_bool_var(f'RoomDayGenderA_in_{r}_on_{d}')

      var_room_day_age = {}
      var_room_day_agediff = {}
      for r in instance.rooms:
        for d in instance.days:
          for age in range(len(instance.age_groups)):
            var_room_day_age[r,d,age] = model.new_bool_var(f'RoomDayAge_{r}_on_{d}_has_{age}')
          var_room_day_agediff[r,d] = model.new_int_var(0, len(instance.age_groups) - 1, f'RoomDayAgediff_{r}_on_{d}_diff_{age}')

      # Every patient is assigned to exactly one room.
      cons_assign = {}
      for g,d in adm_sol.guests_day.items():
        g_data = instance.guests[g]
        if not g_data.is_occupant:
          model.add_exactly_one(var_guest_room[g,r] for r in instance.rooms)

      # Gender restrictions
      for r,r_data in instance.rooms.items():
        for d in instance.days:
          model.add( sum( var_guest_room[g,r] for g in adm_sol.guests_day if d in adm_sol.guest_stay(g) and instance.guests[g].gender == 'A' ) <= r_data.capacity * var_room_day_genderA[r,d] )
          model.add( sum( var_guest_room[g,r] for g in adm_sol.guests_day if d in adm_sol.guest_stay(g) and instance.guests[g].gender == 'B' ) + r_data.capacity * var_room_day_genderA[r,d] <= r_data.capacity )

      # Age indicators
      for key,var in var_guest_room.items():
        g,r = key
        a = instance.age_groups[instance.guests[g].age_group]
        for d in adm_sol.guest_stay(g):
          if isinstance(var, int):
            if var == 1:
              model.add( var_room_day_age[r,d,a] == 1)
          else:
            model.add_implication( var, var_room_day_age[r,d,a] )

      # Age difference
      for r in instance.rooms:
        for d in instance.days:
          for a1 in range(len(instance.age_groups)):
            for a2 in range(a1+1, len(instance.age_groups)):
              model.add( (a2-a1) * (var_room_day_age[r,d,a1] + var_room_day_age[r,d,a2] - 1) <= var_room_day_agediff[r,d] )

      objective = []
      for r in instance.rooms:
        for d in instance.days:
          objective.append( instance.weights['room_mixed_age'] * var_room_day_agediff[r,d] )
      model.Minimize( sum(objective) )

#  model.export_to_file('prob-rooms.txt')

      solver = cp_model.CpSolver()
      solver.parameters.num_workers = 2
#      solver.parameters.log_search_progress = True
      time_limit = min(self._time_limit, algo.remaining_time_parallel())
      solver.parameters.max_time_in_seconds = time_limit
      guests_room_list = []
      solverCallback = ObjectiveSolutionPrinter(instance, algo, guests_room_list, var_guest_room)
      status = solver.solve(model, solverCallback)

      if not guests_room_list:
        adm_sol.set_rooms_infeasible()
        infeas,total = algo.event_room_infeasible(adm_sol)
        print(f'ROOM: CP was infeasible. [{infeas}/{total} infeasible]', flush=True)
      else:
        adm_sol.set_rooms_feasible()
        algo.event_room_feasible(adm_sol)
        infeas,total = algo.get_room_infeasible_counts(adm_sol.room_capacity_reduction)
        print(f'ROOM: CP found {len(guests_room_list)} feasible solutions, out of which {len(guests_room_list[:self._max_num_solutions])} will be considered. [{infeas}/{total} infeasible]', flush=True)

        # Solve theater assignment.
        assignTheaters(instance, algo, adm_sol)

        if adm_sol.is_theaters_infeasible:
          algo.notice_infeasible(adm_sol)
        else:
          guests_room_list.sort()
          print(f'ROOM: Admission solution is {adm_sol}.', flush=True)
          for index,data in enumerate(guests_room_list[:self._max_num_solutions]):
            room_sol = algo.create_room_solution(adm_sol, data[1], index+1, data[2])
            print(f'ROOM: {room_sol}', flush=True)

class GreedyRoomsThread(threading.Thread):

  def __init__(self, instance, algo, max_num_solutions, time_limit, *arg, **kwargs):
    self._instance = instance
    self._algo = algo
    self._time_limit = time_limit
    self._waiting_for_admissions = True
    self._max_num_solutions = max_num_solutions
    assert max_num_solutions > 0
    super().__init__(*arg, **kwargs)

  def signalNoMoreAdmissionSolutions(self):
    self._waiting_for_admissions = False

  def run(self):
    instance = self._instance
    algo = self._algo

    first_sleep = True
    while algo.remaining_time() > 1:
      adm_sol = algo.process_best_admission_solution()
      if adm_sol is None:
        if self._waiting_for_admissions:
          if first_sleep:
            first_sleep = False
            sys.stderr.write(f'AROOM: Sleeping.\n')
            sys.stderr.flush()
          time.sleep(0.1)
          continue
        else:
          sys.stderr.write(f'AROOM: Stopping.\n')
          sys.stderr.flush()
          return

      first_sleep = True

      if adm_sol is not None:
        sys.stderr.write(f'AROOM: Solving heuristic for solution {adm_sol} out of {algo.num_unprocessed_admission_solutions + 1}.\n')
        sys.stderr.flush()

        admitted_patients = list(adm_sol.guests_day.keys())
        admitted_patients.sort(key=lambda p: (adm_sol.guests_day[p]))

        overall_feasible = True
        guests_room = {p : None for p in adm_sol.guests_day}
        room_gender= {r: {d : None for d in instance.days} for r in instance.rooms}
        patients_per_room_per_day = {r: {d : [] for d in instance.days} for r in instance.rooms}


      
          
        
        for o in instance.occupants:
          guests_room[o] = instance.guests[o].room
          for d in adm_sol.guest_stay(o):
              room_gender[instance.guests[o].room][d] = instance.guests[o].gender
              patients_per_room_per_day[instance.guests[o].room][d].append(o)
          admitted_patients.remove(o)
        
               
        while admitted_patients and overall_feasible:
          p = admitted_patients.pop(0)
          compatible_rooms = instance.rooms.keys() - instance.guests[p].incompatible_rooms
          compatible_rooms = list(sorted(compatible_rooms))  
          selected_room = None
          for r in compatible_rooms:
              feasible = True
              for d in adm_sol.guest_stay(p):
                  if room_gender[r][d] != instance.guests[p].gender and room_gender[r][d] is not None:
                      feasible = False
                      break
                  if len(patients_per_room_per_day[r][d]) >= instance.rooms[r].capacity:
                      feasible = False
                      break
              if feasible:
                  selected_room = r
                  break

          if selected_room is None:
              overall_feasible = False
             
          else:
            for d in adm_sol.guest_stay(p):
              patients_per_room_per_day[selected_room][d].append(p)
              room_gender[selected_room][d] = instance.guests[p].gender
            guests_room[p] = selected_room

        
        if overall_feasible:
          adm_sol.set_rooms_feasible()
          algo.event_room_feasible(adm_sol)
          sys.stderr.write(f'AROOM: Room allocation found feasible solution.\n')
          sys.stderr.flush()

          # Solve theater assignment.
          assignTheaters(instance, algo, adm_sol)

          if adm_sol.is_theaters_infeasible:
            algo.notice_infeasible(adm_sol)
          else:
          
            sys.stderr.write(f'AROOM: Admission solution is {adm_sol}.\n')
            print(f"{str(admitted_patients)}")

            room_sol = algo.create_room_solution(adm_sol, guests_room, 0)
            sys.stderr.write(f'AROOM: {room_sol}, time: {self._algo.current_time():.1f}s\n')
            sys.stderr.flush()
        else:
          adm_sol.set_rooms_infeasible()
          infeas,total = algo.event_room_infeasible(adm_sol)
          sys.stderr.write(f'AROOM: Room allocation was infeasible: {infeas}/{total} infeasible.\n')
          sys.stderr.flush()     


class GRASPRoomsThread(threading.Thread):
  
  def __init__(self, instance, algo, max_num_solutions, time_limit, *arg, **kwargs):
    self._instance = instance
    self._algo = algo
    self._time_limit = time_limit
    self._waiting_for_admissions = True
    self._max_num_solutions = max_num_solutions
    assert max_num_solutions > 0
    super().__init__(*arg, **kwargs)

  def signalNoMoreAdmissionSolutions(self):
    self._waiting_for_admissions = False

  def run(self):
    instance = self._instance
    algo = self._algo

    first_sleep = True
    while algo.remaining_time() > 1:
      adm_sol = algo.process_best_admission_solution()
      if adm_sol is None:
        if self._waiting_for_admissions:
          if first_sleep:
            first_sleep = False
            sys.stderr.write(f'AROOM: Sleeping.\n')
            sys.stderr.flush()
          time.sleep(0.1)
          continue
        else:
          sys.stderr.write(f'AROOM: Stopping.\n')
          sys.stderr.flush()
          return

      first_sleep = True

      if adm_sol is not None:
        sys.stderr.write(f'AROOM: Solving heuristic for solution {adm_sol} out of {algo.num_unprocessed_admission_solutions + 1}.\n')
        sys.stderr.flush()
        it = 0
        while it <= 50:
          it +=1
          admitted_patients = list(adm_sol.guests_day.keys())
          random.shuffle(admitted_patients)

          overall_feasible = True
          guests_room = {p : None for p in adm_sol.guests_day}
          room_gender= {r: {d : None for d in instance.days} for r in instance.rooms}
          patients_per_room_per_day = {r: {d : [] for d in instance.days} for r in instance.rooms}


        
            
          
          for o in instance.occupants:
            guests_room[o] = instance.guests[o].room
            for d in adm_sol.guest_stay(o):
                room_gender[instance.guests[o].room][d] = instance.guests[o].gender
                patients_per_room_per_day[instance.guests[o].room][d].append(o)
            admitted_patients.remove(o)
          
        


          while admitted_patients and overall_feasible:
            p = admitted_patients.pop(0)
            compatible_rooms = instance.rooms.keys() - instance.guests[p].incompatible_rooms
            compatible_rooms = list(sorted(compatible_rooms))  
            possible_rooms = []

            for r in compatible_rooms:
                feasible = True
                for d in adm_sol.guest_stay(p):
                    if room_gender[r][d] != instance.guests[p].gender and room_gender[r][d] is not None:
                        feasible = False
                        break
                    if len(patients_per_room_per_day[r][d]) >= instance.rooms[r].capacity:
                        feasible = False
                        break
                if feasible:
                    possible_rooms.append(r)
                    break
            
            selected_room = None
            if len(possible_rooms)>=1:
              possible_rooms.sort(key=lambda r: (patients_per_room_per_day[r][adm_sol.guests_day[p]]), reverse=True)
              selected_room = possible_rooms[0]


            if selected_room is None:
                overall_feasible = False
                
            else:
              for d in adm_sol.guest_stay(p):
                patients_per_room_per_day[selected_room][d].append(p)
                room_gender[selected_room][d] = instance.guests[p].gender
              guests_room[p] = selected_room

          if overall_feasible:
            break


          
        if overall_feasible:
          adm_sol.set_rooms_feasible()
          algo.event_room_feasible(adm_sol)
          
          sys.stderr.write(f'AROOM: Room allocation found feasible solution in GRASP iteration {it}.\n')
          sys.stderr.flush()
          it==5
          # Solve theater assignment.
          assignTheaters(instance, algo, adm_sol)

          if adm_sol.is_theaters_infeasible:
            algo.notice_infeasible(adm_sol)
          else:
            
            sys.stderr.write(f'AROOM: Admission solution is {adm_sol}.\n')
            print(f"{str(admitted_patients)}")

            room_sol = algo.create_room_solution(adm_sol, guests_room, 0)
            sys.stderr.write(f'AROOM: {room_sol}, time: {self._algo.current_time():.1f}s\n')
            sys.stderr.flush()
        else:
          adm_sol.set_rooms_infeasible()
          infeas,total = algo.event_room_infeasible(adm_sol)
          sys.stderr.write(f'AROOM: Room allocation was infeasible: {infeas}/{total} infeasible.\n')
          sys.stderr.flush()     



      





