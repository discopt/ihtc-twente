import sys
import threading

class GuestData:

  def __init__(self, data, is_occupant):
    self._is_occupant = is_occupant
    self._label = data['id']
    self._gender = data['gender']
    self._age_group = data['age_group']
    self._length_of_stay = data['length_of_stay']
    self._workload_produced = data['workload_produced']
    self._skill_level_required = data['skill_level_required']
    self._is_mandatory = data.get('mandatory', True)
    self._surgery_release_day = data.get('surgery_release_day', None)
    self._surgery_due_day = data.get('surgery_due_day', None)
    self._surgery_duration = data.get('surgery_duration', None)
    self._surgeon = data.get('surgeon_id', None)
    self._room = data.get('room_id', None)
    self._incompatible_rooms = data.get('incompatible_room_ids', [])

  @property
  def label(self):
    return self._label

  @property
  def is_occupant(self):
    return self._is_occupant

  @property
  def gender(self):
    return self._gender

  @property
  def age_group(self):
    return self._age_group

  @property
  def length_of_stay(self):
    return self._length_of_stay

  @property
  def workload_produced(self):
    return self._workload_produced

  @property
  def skill_level_required(self):
    return self._skill_level_required

  @property
  def is_mandatory(self):
    return self._is_mandatory

  @property
  def surgery_release_day(self):
    return self._surgery_release_day

  @property
  def surgery_due_day(self):
    return self._surgery_due_day

  @property
  def surgery_duration(self):
    return self._surgery_duration

  @property
  def surgeon(self):
    return self._surgeon

  @property
  def room(self):
    return self._room

  @property
  def incompatible_rooms(self):
    return self._incompatible_rooms

  def __repr__(self):
    if self.is_occupant:
      return f'<occupant of gender {self.gender} staying for {self.length_of_stay} days in room {self.room}>'
    elif self.is_mandatory:
      return f'<mandatory patient of gender {self.gender} admitted for {self.length_of_stay} days within [{self.surgery_release_day},{self.surgery_due_day}]>'
    else:
      return f'<optional patient of gender {self.gender} admitted for {self.length_of_stay} days from day {self.surgery_release_day} on>'

class NurseData:

  def __init__(self, data, shift_types):
    self._label = data['id']
    self._skill_level = data['skill_level']
    self._shifts = {}
    for work in data['working_shifts']:
      self._shifts[3 * work['day'] + shift_types.index(work['shift'])] = work['max_load']

  @property
  def label(self):
    return self._label

  @property
  def skill_level(self):
    return self._skill_level

  @property
  def shifts(self):
    return self._shifts

class RoomData:

  def __init__(self, data):
    self._label = data['id']
    self._capacity = data['capacity']

  @property
  def label(self):
    return self._label

  @property
  def capacity(self):
    return self._capacity

  def __repr__(self):
    return f'<room with capacity {self.capacity}>'

class SurgeonData:

  def __init__(self, data):
    self._label = data['id']
    self._max_surgery_time = data['max_surgery_time']

  @property
  def label(self):
    return self._label

  @property
  def max_surgery_time(self):
    return self._max_surgery_time

class TheaterData:

  def __init__(self, data):
    self._label = data['id']
    self._availability = data['availability']

  @property
  def label(self):
    return self._label

  @property
  def availability(self):
    return self._availability

class Instance:

  def __init__(self, file_name):

    import json
    data = json.loads(open(file_name, 'r').read())

    self._shift_types = tuple(data['shift_types'])
    self._days = tuple(range(data['days']))
    self._skill_levels = data['skill_levels']
    self._shifts = range( len(self._shift_types) * data['days'] )
    self._age_groups = { group: i for i,group in enumerate(data['age_groups']) }
    self._patients = { item['id']: GuestData(item, False) for item in data['patients'] }
    self._occupants = { item['id']: GuestData(item, True) for item in data['occupants'] }
    self._guests = { **self._patients, **self._occupants }
    self._nurses = { item['id']: NurseData(item, self._shift_types) for item in data['nurses'] }
    self._rooms = { item['id']: RoomData(item) for item in data['rooms'] }
    self._surgeons = { item['id']: SurgeonData(item) for item in data['surgeons'] }
    self._theaters = { item['id']: TheaterData(item) for item in data['operating_theaters'] }
    self._genders = ('A', 'B')
    self._weights = data['weights']

    # Calculate total nurse capacities.
    self._total_nurse_capacities = [ 0 ] * self.numShifts
    for n,n_data in self.nurses.items():
      for s,cap in n_data.shifts.items():
        self._total_nurse_capacities[s] += cap


  @property
  def shift_types(self):
    return self._shift_types

  @property
  def numShiftTypes(self):
    return len(self._shift_types)

  @property
  def numDays(self):
    return len(self._days)

  @property
  def days(self):
    return self._days

  @property
  def last_day(self):
    return len(self._days) - 1

  @property
  def numShifts(self):
    return len(self._shifts)

  @property
  def shifts(self):
    return self._shifts

  def shift_to_day(self, shift):
    return shift // len(self._shift_types)

  def day_to_shift(self, day):
    return day * len(self._shift_types)

  def shift_to_type(self, shift):
    return shift % len(self._shift_types)

  def shift_label(self, shift):
    return f'day{self.shift_to_day(shift)}#{self.shift_to_type(shift)}'

  @property
  def skill_levels(self):
    return self._skill_levels

  @property
  def age_groups(self):
    return self._age_groups

  @property
  def patients(self):
    return self._patients

  @property
  def occupants(self):
    return self._occupants

  @property
  def guests(self):
    return self._guests

  @property
  def nurses(self):
    return self._nurses

  @property
  def rooms(self):
    return self._rooms

  @property
  def surgeons(self):
    return self._surgeons

  @property
  def theaters(self):
    return self._theaters

  @property
  def genders(self):
    return self._genders

  @property
  def weights(self):
    return self._weights

  def getTotalNurseCapacities(self):
    return self._total_nurse_capacities

  def __str__(self):
    return f'<instance with {len(self.days)} days = {len(self.shifts)} shifts, {len(self.occupants)} occupants + {len(self.patients)} patients = {len(self.guests)} guests, {len(self.nurses)} nurses, {len(self.surgeons)} surgeons and {len(self.theaters)} theaters; weights are {self._weights}>'

