import random
import sys
from data import *
from algorithm import *

def printUsage():
  sys.stderr.write(f'Usage: {sys.argv[0]} [OPTIONS] INSTANCE-FILE [OUTPUT-FILE]\n')
  sys.stderr.write('Options:\n')
  sys.stderr.write('-Ttotal <NUM> Number of seconds for overall algorithm.\n')
  sys.stderr.write('-Tfinal <NUM> Number of seconds for final phase.\n')
  sys.stderr.write('-Nroom <NUM>  Number of failed room attempts before reducing capacity; default: 6.\n')
  sys.stderr.write('-hlocal       Heuristic shall be local search.\n')
  sys.stderr.write('-hinit        Heuristic shall be initialized from current solution if it exists.\n')
  sys.stderr.flush()

if __name__ == '__main__':
  random.seed(0)

  instance_file_name = None
  output_file_name = None
  length_total = 600
  length_final_phase = 300
  heuristic_local = False
  heuristic_init = False
  num_room_attempts = 6

  i = 1
  while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == '-h' or arg == '--help':
      printUsage()
      sys.exit(0)
    elif arg == '-Ttotal' and i+1 < len(sys.argv):
      length_total = float(sys.argv[i+1])
      i += 1
    elif arg == '-Tfinal' and i+1 < len(sys.argv):
      length_final_phase = float(sys.argv[i+1])
      i += 1
    elif arg == '-Nroom' and i+1 < len(sys.argv):
      num_room_attempts = int(sys.argv[i+1])
      i += 1
    elif arg == '-hlocal':
      heuristic_local = True
    elif arg == '-hinit':
      heuristic_init = True
    elif instance_file_name is None:
      instance_file_name = arg
    elif output_file_name is None:
      output_file_name = arg
    else:
      sys.stderr.write(f'Unexpected argument <{arg}> after <{instance_file_name}> and <{output_file_name}>.\n')
      printUsage()
      sys.exit(1)
    i += 1

  if instance_file_name is None:
    sys.stderr.write(f'Missing INSTANCE argument.\n')
    printUsage()
    sys.exit(1)

  instance = Instance(instance_file_name)
  solve(instance, output_file_name, length_total, length_final_phase, num_room_attempts, heuristic_local, heuristic_init)


