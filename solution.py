import sys
import copy
import threading
import json
from data import *

class AdmissionSolution:
  
  def __init__(self, room_capacity_reduction, patients_day, time):
    '''
    Called by Algorithm to be used to check for duplicates.
    '''
    self._lock = threading.RLock()
    self._room_capacity_reduction = room_capacity_reduction
    self._instance = None
    self._patients_day = patients_day
    self._hash = hash(frozenset(patients_day.items()))
    self._time_admission = time

  def __hash__(self):
    return self._hash

  def __eq__(self, other):
    return self._patients_day == other._patients_day

  def initialize(self, instance, label, patients_days_care_bound):
    '''
    Called by Algorithm class if really new.
    '''
    self._instance = instance
    self._label = label
    
    self._guests_day = { o:0 for o in self._instance.occupants } | self._patients_day

    # Theaters are uninitialized.
    self._theaters_infeasible = None
    self._patients_theater = None
    self._num_open_theaters = None
    self._num_surgeon_transfers = None

    # Room feasibility is uninitialized.
    self._rooms_infeasible = None

    # Some costs are known.
    self._num_unscheduled = sum( 1 for p in self._instance.patients if p not in self._patients_day )
    self._num_delays = 0
    for p,d in self._patients_day.items():
      self._num_delays += d - self._instance.patients[p].surgery_release_day

    # Theater lower bound.
    self._min_open_theaters = 0
    for d in self._instance.days:
      theaters = [ t_data.availability[d] for t_data in self._instance.theaters.values() ]
      theaters.sort()
      total_surgery_duration = sum( self._instance.patients[p].surgery_duration for p,day in self._patients_day.items() if day == d )
      while total_surgery_duration > 0:
        assert theaters
        self._min_open_theaters += 1
        total_surgery_duration -= theaters.pop()

    # Nurse excess workload lower bound.
    nurse_excess = {}
    for n,n_data in self._instance.nurses.items():
      for s,cap in n_data.shifts.items():
        nurse_excess[s] = nurse_excess.get(s, 0) - cap
    for g in self.guests:
      for s,workload in self.guest_shift_workload(g).items():
        nurse_excess[s] = nurse_excess.get(s, 0) + workload
    self._min_nurse_workload = 0
    for s,excess in nurse_excess.items():
      if excess > 0:
        self._min_nurse_workload += excess

    # Care cost bound.
    self.update_bound_care(patients_days_care_bound)

  def update_bound_care(self, patients_days_care_bound):
    self._bound_care = 0
    for p,d in self._patients_day.items():
      self._bound_care += patients_days_care_bound[p,d]

  @property
  def instance(self):
    return self._instance

  @property
  def room_capacity_reduction(self):
    return self._room_capacity_reduction

  @property
  def time_admission(self):
    return self._time_admission

  @property
  def is_theaters_infeasible(self):
    return self._theaters_infeasible

  def set_theaters_infeasible(self):
    self._theaters_infeasible = True

  def pre_theaters(self):
    self._theaters_infeasible = False
    self._patients_theater = {}

  def set_theater(self, p, t):
    with self._lock:
      assert self._patients_theater is not None
      self._patients_theater[p] = t

  def post_theaters(self):
    count_open_theaters = 0
    self._num_surgeon_transfers = 0
    for d in self._instance.days:
      theater_patients = { t:[] for t in self._instance.theaters }
      surgeon_theaters = { s:set() for s in self._instance.surgeons }
      for p,t in self._patients_theater.items():
        if self._patients_day[p] == d:
          theater_patients[t].append(p)
          surgeon_theaters[self._instance.patients[p].surgeon].add(t)
      for t,ps in theater_patients.items():
        if ps:
          count_open_theaters += 1
      for s,ts in surgeon_theaters.items():
        if len(ts) > 1:
          self._num_surgeon_transfers += len(ts) - 1
    self._num_open_theaters = count_open_theaters

  def set_rooms_infeasible(self):
    self._rooms_infeasible = True

  def set_rooms_feasible(self):
    self._rooms_infeasible = False

  def __repr__(self):
    s = f'Adm#{self._label}: '
    s += f'${self.costs_unscheduled} unsched'
    s += f' + ${self.costs_delays} delays'
    if self.has_theaters:
      s += f' + ${self.costs_open_theaters} thea'
      s += f' + ${self.costs_surgeon_transfers} surg'
    else:
      s += f' + >${self.bound_open_theaters} thea'
    s += f' + >${self.bound_nurse_workload} wkld'
    s += f' + >${self.bound_care} care'
    s += f' = ${self.costs_total} + >${self.bound_extra} > ${self.bound_total}'
    s += f' [RC={self.room_capacity_reduction},t_A={self._time_admission:.1f}s]'
    return s

  @property
  def patients_day(self):
    return self._patients_day

  @property
  def label(self):
    return self._label

  @property
  def guests(self):
    return self._guests_day.keys()

  @property
  def guests_day(self):
    return self._guests_day

  def guest_stay(self, g):
    '''
    Return range of days in which the guest stays.
    '''
    d_first = self._guests_day[g]
    d_beyond = min(self._instance.numDays, d_first + self._instance.guests[g].length_of_stay)
    return range(d_first, d_beyond)

  def guest_shift_workload(self, g):
    '''
    Returns the shifts in which the guest stays, along with the produced workload
    '''
    g_data = self._instance.guests[g]
    s_first = 3 * self._guests_day[g]
    s_beyond = min(s_first + 3 * g_data.length_of_stay, self._instance.numShifts)
    return { s:g_data.workload_produced[s - s_first] for s in range(s_first, s_beyond) }
  
  @property
  def num_unscheduled(self):
    return self._num_unscheduled

  @property
  def costs_unscheduled(self):
    return self._instance.weights['unscheduled_optional'] * self.num_unscheduled

  @property
  def num_delays(self):
    return self._num_delays

  @property
  def costs_delays(self):
    return self._instance.weights['patient_delay'] * self.num_delays

  @property
  def has_theaters(self):
    return self._num_open_theaters is not None

  @property
  def patients_theater(self):
    return self._patients_theater

  @property
  def num_open_theaters(self):
    return self._num_open_theaters

  @property
  def costs_open_theaters(self):
    assert self.has_theaters
    return self._instance.weights['open_operating_theater'] * self.num_open_theaters

  @property
  def num_surgeon_transfers(self):
    return self._num_surgeon_transfers

  @property
  def costs_surgeon_transfers(self):
    return self._instance.weights['surgeon_transfer'] * self.num_surgeon_transfers

  @property
  def min_open_theaters(self):
    return self._min_open_theaters

  @property
  def bound_open_theaters(self):
    return self._instance.weights['open_operating_theater'] * self.min_open_theaters

  @property
  def min_nurse_workload(self):
    return self._min_nurse_workload

  @property
  def bound_nurse_workload(self):
    return self._instance.weights['nurse_eccessive_workload'] * self.min_nurse_workload

  @property
  def bound_care(self):
    return self._bound_care

  @property
  def costs_total(self):
    if self.has_theaters:
      return self.costs_unscheduled + self.costs_delays + self.costs_open_theaters + self.costs_surgeon_transfers
    else:
      return self.costs_unscheduled + self.costs_delays

  @property
  def bound_extra(self):
    if self.has_theaters:
      return self.bound_nurse_workload + self.bound_care
    else:
      return self.bound_nurse_workload + self.bound_care + self.bound_open_theaters

  @property
  def bound_total(self):
    return self.costs_total + self.bound_extra


class RoomSolution:
  
  def __init__(self, adm_sol, guests_room, time_room, time_room_relative):
    self._adm_sol = adm_sol
    self._guests_room = guests_room
    self._hash = hash((adm_sol, frozenset(guests_room.items())))
    self._time_room = time_room
    self._time_room_relative = time_room_relative
    self._time_nurse = None
    self._shift_min_cost = None
    self._nurses_min_cost = None

  def __hash__(self):
    return self._hash

  def __eq__(self, other):
    return self._adm_sol is other._adm_sol and self._guests_room == other._guests_room

  def initialize(self, label):
    self._label = label
    self._room_shift_nurse = None
    
    # Compute mapping from room/day pairs to list of guests.
    self._room_day_guests = { (r,d):[] for r in self.instance.rooms for d in self.instance.days }
    for g in self._adm_sol.guests_day:
      for d in self._adm_sol.guest_stay(g):
        
        self._room_day_guests[self._guests_room[g], d].append(g)

    # For each room/day pair, check the maximum and minimum ages.
    self._num_agemix = 0
    for key,gs in self._room_day_guests.items():
      if len(gs) >= 2:
        min_age = min( self.instance.age_groups[self.instance.guests[g].age_group] for g in gs )
        max_age = max( self.instance.age_groups[self.instance.guests[g].age_group] for g in gs )
        self._num_agemix += max_age - min_age

  def set_room_shift_nurse(self, room_shift_nurse, time_nurse):
    instance = self.instance
    self._room_shift_nurse = room_shift_nurse
    self._time_nurse = time_nurse

    # Excess workload.
    nurse_shift_capacity = { (n,s):cap for n,n_data in instance.nurses.items() for s,cap in n_data.shifts.items() }
    nurse_shift_work = { key:0 for key in nurse_shift_capacity }

    # Skill level
    self._num_nurse_skill_deviations = 0
    for key,n in room_shift_nurse.items():
      r,s = key
      d = instance.shift_to_day(s)
      for g in self._room_day_guests[r,d]:
        g_data = instance.guests[g]
        required = g_data.skill_level_required[s - 3*self.guests_day[g]]
#        print(f'In {r} at {s}, guest {g} stays and is served by {n}; required skill level {required}, provided is {instance.nurses[n].skill_level}')
        self._num_nurse_skill_deviations += max(0, required - instance.nurses[n].skill_level)

    # Continuity of care.
    self._num_nurse_continuity_of_care = 0
    for g,r in self.guests_room.items():
      nurses = set()
      for d in self.guest_stay(g):
        for s in [3*d, 3*d+1, 3*d+2]:
          n = room_shift_nurse[r,s]
          nurses.add(n)
#          print(f'In {r} at {s}, guest {g} stays and produces workload of {instance.guests[g].workload_produced[s - 3*self.guests_day[g]]}')
          nurse_shift_work[n,s] += instance.guests[g].workload_produced[s - 3*self.guests_day[g]]
#      print(f'Guest {g} in {r} is served by {nurses}')
      self._num_nurse_continuity_of_care += len(nurses)

    self._num_nurse_workload = 0
    for key,work in nurse_shift_work.items():
      cap = nurse_shift_capacity[key]
#      print(f'Nurse {key[0]} at shift {key[1]} has work {work} out of {cap}')
      self._num_nurse_workload += max(0, work - cap)

  def set_shift_min_cost(self, shift_min_cost):
    self._shift_min_cost = shift_min_cost

  def set_nurses_min_cost(self, nurses_min_cost):
    self._nurses_min_cost = nurses_min_cost

  def save(self, file_name):
    f = open(file_name, 'w')
    output = { 'patients': [], 'nurses': [] }
    for p,d in self.patients_day.items():
      output['patients'].append( { 'id': p, 'admission_day': d, 'room': self.guests_room[p] , 'operating_theater': self.patients_theater[p] } )
    for n,n_data in self.instance.nurses.items():
      output['nurses'].append( {'id': n, 'assignments': []} ) 
      for s in n_data.shifts:
        rooms = []
        for r in self.instance.rooms:
          if self.room_shift_nurse.get((r,s), None) == n:
            rooms.append( r )
        output['nurses'][-1]['assignments'].append( { 'day': self.instance.shift_to_day(s), 'shift': self.instance.shift_types[self.instance.shift_to_type(s)], 'rooms': rooms } )
        
    f.write(json.dumps(output, indent=2))
    f.close()

    

  def __repr__(self):
    if self._room_shift_nurse is None:
      s = f'Room#{self._label}: '
      s += f'${self.costs_unscheduled} unsched'
      s += f' + ${self.costs_delays} delays'
      s += f' + ${self.costs_open_theaters} thea'
      s += f' + ${self.costs_surgeon_transfers} surg'
      s += f' + ${self.costs_agemix} agmx'
      s += f' + >${self.bound_nurse_workload} wkld'
      s += f' + >${self.bound_care} care'
      s += f' = ${self.costs_total} + >${self.bound_extra} > ${self.bound_total}'
      s += f' [RC={self.room_capacity_reduction},t_A={self.time_admission:.1f}s,t_R={self._time_room:.1f}s,t_r={self._time_room_relative:.1f}s]'
    else:
      s = f'Sol#{self._label}: '
      s += f'${self.costs_unscheduled} unsched'
      s += f' + ${self.costs_delays} delays'
      s += f' + ${self.costs_open_theaters} thea'
      s += f' + ${self.costs_surgeon_transfers} surg'
      s += f' + ${self.costs_agemix} agmx'
      s += f' + ${self.costs_nurse_workload} wkld'
      s += f' + ${self.costs_continuity_care} coc'
      s += f' + ${self.costs_skill} skill'
      s += f' = ${self.costs_total}'
      if self._nurses_min_cost is not None:
        s += f' > ${self._nurses_min_cost}'
      s += f' [RC={self.room_capacity_reduction},t_A={self.time_admission:.1f}s,t_R={self._time_room:.1f}s,t_r={self._time_room_relative:.1f}s,t_N={self._time_nurse:.1f}s]'

    return s

  @property
  def time_admission(self):
    return self._adm_sol.time_admission
  
  @property
  def patients_day(self):
    return self._adm_sol.patients_day

  @property
  def has_nurses(self):
    return self._room_shift_nurse is not None

  @property
  def instance(self):
    return self._adm_sol.instance

  @property
  def guests_day(self):
    return self._adm_sol.guests_day

  def guest_stay(self, g):
    return self._adm_sol.guest_stay(g)

  @property
  def guests_room(self):
    return self._guests_room

  @property
  def room_day_guests(self):
    return self._room_day_guests

  @property
  def room_shift_nurse(self):
    return self._room_shift_nurse

  @property
  def costs_unscheduled(self):
    return self._adm_sol.costs_unscheduled

  @property
  def costs_delays(self):
    return self._adm_sol.costs_delays

  @property
  def costs_open_theaters(self):
    return self._adm_sol.costs_open_theaters

  @property
  def costs_surgeon_transfers(self):
    return self._adm_sol.costs_surgeon_transfers

  @property
  def room_capacity_reduction(self):
    return self._adm_sol.room_capacity_reduction

  @property
  def label(self):
    return self._label

  @property
  def num_agemix(self):
    return self._num_agemix

  @property
  def costs_agemix(self):
    return self.instance.weights['room_mixed_age'] * self.num_agemix

  @property
  def costs_nurse_workload(self):
    return self.instance.weights['nurse_eccessive_workload'] * self._num_nurse_workload

  @property
  def costs_continuity_care(self):
    return self.instance.weights['continuity_of_care'] * self._num_nurse_continuity_of_care

  @property
  def costs_skill(self):
    return self.instance.weights['room_nurse_skill'] * self._num_nurse_skill_deviations

  @property
  def bound_nurse_workload(self):
    return self._adm_sol.bound_nurse_workload

  @property
  def bound_care(self):
    return self._adm_sol.bound_care

  @property
  def costs_no_nurse(self):
    return self._adm_sol.costs_total + self.costs_agemix

  @property
  def costs_total(self):
    if self.has_nurses:
      return self.costs_no_nurse + self.costs_nurse_workload + self.costs_continuity_care + self.costs_skill
    else:
      return self.costs_no_nurse

  @property
  def patients_theater(self):
    return self._adm_sol.patients_theater

  @property
  def bound_extra(self):
    return self._adm_sol.bound_extra

  @property
  def bound_total(self):
    return self._adm_sol.bound_total + self.costs_agemix

  @property
  def has_shift_min_cost(self):
    return self._shift_min_cost is not None

  @property
  def shift_min_cost(self):
    return self._shift_min_cost

