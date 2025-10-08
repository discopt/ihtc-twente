from data import *

for instance_file_name in sys.argv[1:]:
  instance = Instance(instance_file_name)
  print(instance_file_name, instance)
