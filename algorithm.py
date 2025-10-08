import time
import threading
import queue
from data import *
from nurses import *
from care import *
from admissions import *
from rooms import *
from solution import *
from final import *

class Algorithm:

  def __init__(self, instance, output_file_name, max_room_infeasible_count, max_num_incumbents, time_limit_parallel, time_limit_total):
    self._instance = instance
    self._output_file_name = output_file_name
    self._time_started = time.time()
    self._time_limit_parallel = float(time_limit_parallel)
    self._time_limit_total = float(time_limit_total)

    # We stop generating admission solutions if at least max_room_infeasible_count *and* at least 80% failed.
    self._max_room_infeasible_count = max_room_infeasible_count
    self._max_room_infeasible_ratio = 0.9

    # Admission solution storage.
    self._admission_solutions_data = {}
    self._data_admission_solution = {}
    self._unprocessed_admission_solutions = []
    self._admission_solutions_lock = threading.RLock()

    # Room solution storage.
    self._room_solutions_data = {}
    self._data_room_solution = {}
    self._unprocessed_room_solutions = []
    self._room_solutions_lock = threading.RLock()

    # Room capacity reduction.
    self._admission_thread = None
    self._room_capacity_reduction = 0
    self._room_infeasible_counts = {} # reduction -> (number of infeasible solutions (0 if absent), number of considered solutions)

    # Incumbent
    self._incumbents = []
    self._max_num_incumbents = max_num_incumbents
    self._final_incumbent = None
    self._final_incumbent_cost = float('inf')

    # Lower bound
    self._best_lower_bound = 0.0

  def create_admission_solution(self, room_capacity_reduction, patients_day, patients_days_care_bound):
    with self._admission_solutions_lock:
      data = AdmissionSolution(room_capacity_reduction, patients_day, self.current_time())
      sol = self._data_admission_solution.get(data, None)
      if sol is None:
        sol = f'{len(self._admission_solutions_data) + 1:03d}'
        self._admission_solutions_data[sol] = data
        self._data_admission_solution[data] = sol
        data.initialize(self._instance, sol, patients_days_care_bound)
        self._unprocessed_admission_solutions.append(sol)
      else:
        data = self._admission_solutions_data[sol]
      return data

  def find_best_admission_solution(self):
    # TODO: Add room_capacity_reduction parameter to filter out only relevant solutions.
    with self._admission_solutions_lock:
      min_bound = float('inf')
      min_data = None
      for sol,data in self._admission_solutions_data.items():
        if data.bound_total < min_bound:
          min_bound = data.bound_total
          min_data = data
      return min_data

  def process_best_admission_solution(self):
    with self._admission_solutions_lock:
      min_bound = float('inf')
      min_sol = None
      min_data = None
      for sol in self._unprocessed_admission_solutions:
        data = self._admission_solutions_data[sol]
        bound = data.bound_total
        infeas,total = self._room_infeasible_counts.get(data.room_capacity_reduction, (0,0))
        if infeas >= self._max_room_infeasible_count and (infeas * 1.0 / total) >= self._max_room_infeasible_ratio:
          bound += 1_000_000
        if bound < min_bound:
          min_bound = bound
          min_sol = sol
          min_data = data
      if min_sol is not None:
        self._unprocessed_admission_solutions.remove(min_sol)
      return min_data

  def create_room_solution(self, adm_sol, guests_room, index, time_room_relative):
    with self._room_solutions_lock:
      data = RoomSolution(adm_sol, guests_room, self.current_time(), time_room_relative)
      sol = self._data_room_solution.get(data, None)
      if sol is None:
        sol = f'{adm_sol.label}.{index:02d}'
        self._room_solutions_data[sol] = data
        self._data_room_solution[data] = sol
        data.initialize(sol)
        self._unprocessed_room_solutions.append(sol)
      else:
        data = self._room_solutions_data[sol]
      return data

  def set_admission_thread(self, thread):
    self._admission_thread = thread

  def event_room_feasible(self, adm_sol):
    infeas,total = self._room_infeasible_counts.get(adm_sol.room_capacity_reduction, (0, 0))
    total += 1
    self._room_infeasible_counts[adm_sol.room_capacity_reduction] = (infeas, total)

  def get_room_infeasible_counts(self, room_capacity_reduction):
    return self._room_infeasible_counts.get(room_capacity_reduction, (0, 0))

  def event_room_infeasible(self, adm_sol):
    infeas,total = self._room_infeasible_counts.get(adm_sol.room_capacity_reduction, (0, 0))
    infeas += 1
    total += 1
    self._room_infeasible_counts[adm_sol.room_capacity_reduction] = (infeas, total)
    if infeas >= self._max_room_infeasible_count and (infeas * 1.0 / total) > self._max_room_infeasible_ratio and self._admission_thread is not None and self._admission_thread.room_capacity_reduction == adm_sol.room_capacity_reduction:
      print(f'DISP: Interrupting admission solution thread due to {infeas} / {total} failures.', flush=True)
      try:
        self._admission_thread.interrupt()
      except AttributeError:
        pass
    return infeas,total
  
  def find_best_room_solution(self):
    with self._room_solutions_lock:
      min_bound = float('inf')
      min_data = None
      for sol,data in self._room_solutions_data.items():
        if data.bound_total < min_bound:
          min_bound = data.bound_total
          min_data = data
      return min_data

  def process_best_room_solution(self):
    with self._room_solutions_lock:
      min_bound = float('inf')
      min_sol = None
      min_data = None
      for sol in self._unprocessed_room_solutions:
        data = self._room_solutions_data[sol]
        bound = data.bound_total
        if bound < min_bound:
          min_bound = bound
          min_sol = sol
          min_data = data
      if min_sol is not None:
        self._unprocessed_room_solutions.remove(min_sol)
      return min_data

  @property
  def best_lower_bound(self):
    return self._best_lower_bound

  @property
  def incumbents(self):
    return self._incumbents

  @property
  def final_incumbent(self):
    return self._final_incumbent

  def event_new_solution(self, sol):
    if sol not in self._incumbents:
      self._incumbents.append(sol)
    self._incumbents.sort(key=lambda s: s.costs_total)
    if len(self._incumbents) > self._max_num_incumbents:
      del self._incumbents[-1]

    self.event_new_final_solution(sol)

  def event_new_final_solution(self, sol):
    if self._final_incumbent is None or sol.costs_total < self._final_incumbent_cost:
      self._final_incumbent = sol
      self._final_incumbent_cost = sol.costs_total
      print(f'\n!!! {sol}\n', flush=True)
      if self._output_file_name is not None:
        self._final_incumbent.save(self._output_file_name)

  def current_time(self):
    return float(time.time() - self._time_started)

  def remaining_time_parallel(self):
    return self._time_limit_parallel - self.current_time()

  def remaining_time_total(self):
    return self._time_limit_total - self.current_time()

  @property
  def final_incumbent_cost(self):
    return self._final_incumbent_cost

  @property
  def instance(self):
    return self._instance

  @property
  def num_unprocessed_admission_solutions(self):
    return len(self._unprocessed_admission_solutions)

  @property
  def num_unprocessed_room_solutions(self):
    return len(self._unprocessed_room_solutions)

  @property
  def admission_solutions(self):
    return self._admission_solutions_data


class CareCostsThread(threading.Thread):

  def __init__(self, queue, *arg, **kwargs):
    self._queue = queue
    super().__init__(*arg, **kwargs)

  def run(self):
    while True:
      try:
        task = self._queue.get(timeout=0.01)
      except queue.Empty:
        return
      try:
        task.run()
      except Exception as e:
        print(e)
      self._queue.task_done()

def solve(instance, output_file_name, length_total, length_final_phase, num_room_attempts, heuristic_local, heuristic_init):
  num_care_cost_threads = 3
  num_final_threads = 4
  rooms_time_limit = 5.0
  rooms_max_num_solutions = 10
  nurses_time_limit = 15.0
  max_num_incumbents = 100

  algo = Algorithm(instance, output_file_name, num_room_attempts, max_num_incumbents, length_total - length_final_phase, length_total)

  # Dictionary for lower bound on care costs for patient/day combinations.
  patients_days_care_bound = {}

  care_cost_queue = queue.Queue()
  for p,p_data in instance.patients.items():
    costs_unscheduled = instance.weights['unscheduled_optional']

    d_first = p_data.surgery_release_day
    d_beyond = p_data.surgery_due_day + 1 if p_data.is_mandatory else instance.last_day + 1
    for d in range(d_first, d_beyond):

      # Surgeon has no time.
      if p_data.surgery_duration > instance.surgeons[p_data.surgeon].max_surgery_time[d]:
        continue

      costs_delay = instance.weights['patient_delay'] * (d - p_data.surgery_release_day)
      costs_care_trivial = instance.numShiftTypes * instance.weights['continuity_of_care']

      # Not scheduling is as good.
      if not p_data.is_mandatory and costs_delay + costs_care_trivial >= costs_unscheduled:
        continue

      patients_days_care_bound[p,d] = costs_care_trivial
      care_cost_queue.put_nowait( CareCostsTask(instance, p, d, patients_days_care_bound) )

  print(f'DISP: Computing lower bounds on care costs for {care_cost_queue.qsize()} patient/day combinations using {num_care_cost_threads} threads.', flush=True)

  # Start thread for patient-day assignment without care costs.
  admissions_thread = AdmissionsThread(instance, patients_days_care_bound, 0, 'i', algo, algo.remaining_time_parallel())
  admissions_thread.start()

  # Start threads for care costs.
  for _ in range(num_care_cost_threads):
    CareCostsThread(care_cost_queue).start()

  print(f'DISP: Waiting for {num_care_cost_threads} care-cost threads.', flush=True)
  care_cost_queue.join()
  admissions_thread.interrupt()

  print(f'DISP: Interrupting admission thread.', flush=True)
  admissions_thread.join()

  print(f'\nSTARTING PHASE 2 after {algo.current_time():.1f}s\n', flush=True)

  print(f'DISP: Updating care-costs for {len(algo.admission_solutions)} admission solutions.', flush=True)
  for sol in algo.admission_solutions.values():
    sol.update_bound_care(patients_days_care_bound)

  print(f'DISP: Starting room thread.', flush=True)

  roomsThread = RoomsThread(instance, algo, rooms_max_num_solutions, rooms_time_limit)
  roomsThread.start()

  print(f'DISP: Starting nurse thread.', flush=True)

  nursesThread = NursesThread(instance, algo, nurses_time_limit, heuristic_local, heuristic_init)
  nursesThread.start()

  # Start thread for patient-day assignment with care costs.
  for room_capacity_reduction in range(10):
    print(f'DISP: Starting admission thread with room capacity reduced by {room_capacity_reduction}.', flush=True)

    admissions_thread = AdmissionsThread(instance, patients_days_care_bound, room_capacity_reduction, str(room_capacity_reduction), algo, algo.remaining_time_parallel(), initial_solution=algo.find_best_admission_solution())
    algo.set_admission_thread(admissions_thread)
    admissions_thread.start()
    admissions_thread.join()
    print(f'DISP: Stopped admission thread with room capacity reduced by {room_capacity_reduction}.', flush=True)

    algo.set_admission_thread(None)
    if algo.remaining_time_parallel() < 1:
      break

  roomsThread.signalNoMoreAdmissionSolutions()
  roomsThread.join()
  nursesThread.signalNoMoreRoomSolutions()
  nursesThread.join()
  
  print(f'Best {len(algo.incumbents)} solutions:')
  for sol in algo.incumbents:
    print(f'{sol}')

  print(f'\nSTARTING PHASE 3 after {algo.current_time():.1f}s\n', flush=True)

  final_queue = queue.Queue()
  for sol in algo.incumbents:
    final_queue.put_nowait( FinalNursesTask(sol) )

  # Start threads for final solutions.
  for thread_index in range(num_final_threads):
    FinalNursesThread(thread_index, final_queue, instance, algo).start()

  print(f'DISP: Waiting for {num_final_threads} final-nurses threads.', flush=True)
  final_queue.join()

  print(f'\nTotal time: {algo.current_time():.1f}s')
  print(f'Final sol: {algo.final_incumbent}', flush=True)
  print(f'Final lower bound: {algo.best_lower_bound}', flush=True)

