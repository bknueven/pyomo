from pyomo.environ import *

ipopt = SolverFactory('ipopt')
ipopt = ipopt.available()

print(f'ipopt available {ipopt}')
