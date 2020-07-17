#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import re
import sys
import time
from pyutilib.misc import Bunch
from pyutilib.services import TempfileManager
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import DirectOrPersistentSolver
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
import pyomo.core.base.var


logger = logging.getLogger('pyomo.solvers')

class DegreeError(ValueError):
    pass

class _XpressExpr(object):
    def __init__(self, constant=0, linear_vars=None, linear_coefs=None,
            quad_vars_1=None, quad_vars_2=None, quad_coefs=None):
        self.constant = constant
        self.linear_vars = linear_vars
        self.linear_coefs = linear_coefs 
        self.quad_vars_1 = quad_vars_1
        self.quad_vars_2 = quad_vars_2
        self.quad_coefs = quad_coefs

class _VariableData(object):
    def __init__(self):
        self.lb = list()
        self.ub = list()
        self.types = list()
        self.names = list()

        self.chgbounds_vals = list()
        self.chgbounds_type = list()
        self.chgbounds_idx = list()

    def add(self, lb, ub, vartype, name, idx):
        self.lb.append(lb)
        self.ub.append(ub)
        self.types.append(vartype)
        self.names.append(name)

        if vartype == 'B' and (lb > 0 or ub < 1):
            if lb == ub:
                self.chgbounds_vals.append(lb)
                self.chgbounds_type.append('B')
                self.chgbounds_idx.append(idx)
            else:
                self.chgbounds_vals.append(lb)
                self.chgbounds_vals.append(ub)
                self.chgbounds_type.append('L')
                self.chgbounds_type.append('U')
                self.chgbounds_idx.append(idx)
                self.chgbounds_idx.append(idx)

    def store_in_xpress(self, solver_model):
        objx = [0.]*len(self.types)
        mstart = [0]*(len(self.types)+1)
        solver_model.addcols(objx=objx, mstart=mstart, mrwind=[], dmatval=[],
                                    bdl=self.lb, bdu=self.ub, names=self.names, types=self.types)

        ## bounds on binary variables don't seem to be set correctly
        ## by the method above
        if len(self.chgbounds_vals) > 0:
            solver_model.chgbounds(self.chgbounds_idx, self.chgbounds_type, self.chgbounds_vals)


class _ConstraintData(object):
    def __init__(self, solver_model):
        self._solver_model = solver_model
        self.sense = list()
        self.rhs = list()
        self.mstart = [0]
        self.mclind = list()
        self.dmatval = list()
        self.range = list()
        self.names = list()

        self.q_idxs = list()
        self.q_vars1 = list()
        self.q_vars2 = list()
        self.q_coefs = list()

    def add(self, sense, rhs, linear_vars, linear_coefs, conname, x_range=0.,
            q_idx=None, q_var1=None, q_var2=None, q_coef=None):
        self.sense.append(sense)
        self.rhs.append(rhs)
        self.mclind.extend(linear_vars)
        self.dmatval.extend(linear_coefs)
        self.names.append(conname)
        self.range.append(x_range)
        self.mstart.append(len(self.mclind))

        if q_idx is not None:
            self.q_idxs.append(q_idx)
            self.q_vars1.append(q_var1)
            self.q_vars2.append(q_var2)
            self.q_coefs.append(q_coef)

    def store_in_xpress(self):
        if len(self.mstart) <= 1:
            return
        solver_model = self._solver_model
        solver_model.addrows(self.sense, self.rhs, self.mstart, self.mclind,
                self.dmatval, range=self.range, names=self.names)
        for idx, var1, var2, coef in zip(self.q_idxs, self.q_vars1, self.q_vars2, self.q_coefs):
            solver_model.addqmatrix(idx, var1, var2, coef)

def _is_convertable(conv_type,x):
    try:
        conv_type(x)
    except ValueError:
        return False
    return True

def _print_message(xp_prob, _, msg, *args):
    if msg is not None:
        sys.stdout.write(msg+'\n')
        sys.stdout.flush()

@SolverFactory.register('xpress_direct', doc='Direct python interface to XPRESS')
class XpressDirect(DirectSolver):

    def __init__(self, **kwds):
        if 'type' not in kwds:
            kwds['type'] = 'xpress_direct'
        super(XpressDirect, self).__init__(**kwds)
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()

        self._pyomo_var_to_var_idx_map = ComponentMap()
        self._var_idx_count = 0
        self._con_idx_count = 0

        self._pyomo_sos_to_sos_idx_map = dict()
        self._sos_idx_count = 0

        self._name = None
        try:
            import xpress 
            self._xpress = xpress
            self._python_api_exists = True
            self._version = tuple(
                    int(k) for k in self._xpress.getversion().split('.'))
            self._name = "Xpress %s.%s.%s" % self._version
            self._version_major = self._version[0]
            # in versions prior to 34, xpress raised a RuntimeError, but in more
            # recent versions it raises a xpress.ModelError. We'll cache the appropriate
            # one here
            if self._version_major < 34:
                self._XpressException = RuntimeError
            else:
                self._XpressException = xpress.ModelError
        except ImportError:
            self._python_api_exists = False
        except Exception as e:
            # other forms of exceptions can be thrown by the xpress python
            # import. for example, a xpress.InterfaceError exception is thrown
            # if the Xpress license is not valid. Unfortunately, you can't
            # import without a license, which means we can't test for the
            # exception above!
            print("Import of xpress failed - xpress message=" + str(e) + "\n")
            self._python_api_exists = False
            
        self._range_constraints = dict()

        # TODO: this isn't a limit of XPRESS, which implements an SLP
        #       method for NLPs. But it is a limit of *this* interface
        self._max_obj_degree = 2
        self._max_constraint_degree = 2

        # There does not seem to be an easy way to get the
        # wallclock time out of xpress, so we will measure it
        # ourselves
        self._opt_time = None

        # Note: Undefined capabilites default to None
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _apply_solver(self):
        if not self._save_results:
            for block in self._pyomo_model.block_data_objects(descend_into=True,
                                                              active=True):
                for var in block.component_data_objects(ctype=pyomo.core.base.var.Var,
                                                        descend_into=False,
                                                        active=True,
                                                        sort=False):
                    var.stale = True

        self._solver_model.setlogfile(self._log_file)
        if self._keepfiles:
            print("Solver log file: "+self.log_file)

        # setting a log file in xpress disables all output
        # this callback prints all messages to stdout
        if self._tee:
            self._solver_model.addcbmessage(_print_message, None, 0)

        # set xpress options
        # if the user specifies a 'mipgap', set it, and
        # set xpress's related options to 0.
        if self.options.mipgap is not None:
            self._solver_model.setControl('miprelstop', float(self.options.mipgap))
            self._solver_model.setControl('miprelcutoff', 0.0)
            self._solver_model.setControl('mipaddcutoff', 0.0)
        # xpress is picky about the type which is passed
        # into a control. So we will infer and cast
        # get the xpress valid controls
        xp_controls = self._xpress.controls
        for key, option in self.options.items():
            if key == 'mipgap': # handled above
                continue
            try: 
                self._solver_model.setControl(key, option)
            except self._XpressException:
                # take another try, converting to its type
                # we'll wrap this in a function to raise the
                # xpress error
                contr_type = type(getattr(xp_controls, key))
                if not _is_convertable(contr_type, option):
                    raise
                self._solver_model.setControl(key, contr_type(option))

        start_time = time.time()
        self._solver_model.solve()
        self._opt_time = time.time() - start_time

        self._solver_model.setlogfile('')
        if self._tee:
            self._solver_model.removecbmessage(_print_message, None)

        # FIXME: can we get a return code indicating if XPRESS had a significant failure?
        return Bunch(rc=None, log=None)

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2, objective=False):
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > max_degree):
            raise DegreeError('XpressDirect does not support expressions of degree {0}.'.format(degree))

        # NOTE: xpress's python interface only allows for expresions
        #       with native numeric types. Others, like numpy.float64,
        #       will cause an exception when constructing expressions
        pyomo_var_to_var_idx_map = self._pyomo_var_to_var_idx_map
        x_linear_vars = [pyomo_var_to_var_idx_map[var] for var in repn.linear_vars]
        x_linear_coefs = [float(coef) for coef in repn.linear_coefs]
        x_const = float(repn.constant)

        if len(repn.linear_vars) > 0:
            referenced_vars.update(repn.linear_vars)

        if len(repn.quadratic_coefs) > 0:
            x_quad_var_1 = list()
            x_quad_var_2 = list()
            x_quad_coefs = list()
            if not objective:
                for coef,(x,y) in zip(repn.quadratic_coefs,repn.quadratic_vars):
                    idx_1 = pyomo_var_to_var_idx_map[x]
                    idx_2 = pyomo_var_to_var_idx_map[y]
                    x_quad_var_1.append(idx_1)
                    x_quad_var_2.append(idx_2)
                    referenced_vars.add(x)
                    referenced_vars.add(y)

                    # NOTE: xpress wants only half of the triangle,
                    #       so these need to be adjusted
                    if idx_1 == idx_2:
                        x_quad_coefs.append(float(coef))
                    else:
                        x_quad_coefs.append(float(coef)/2.)
            else:
                for coef,(x,y) in zip(repn.quadratic_coefs,repn.quadratic_vars):
                    idx_1 = pyomo_var_to_var_idx_map[x]
                    idx_2 = pyomo_var_to_var_idx_map[y]
                    x_quad_var_1.append(idx_1)
                    x_quad_var_2.append(idx_2)
                    referenced_vars.add(x)
                    referenced_vars.add(y)

                    # NOTE: xpress wants only half of the triangle,
                    #       so these need to be adjusted
                    if idx_1 == idx_2:
                        x_quad_coefs.append(2.*float(coef))
                    else:
                        x_quad_coefs.append(float(coef))

            new_expr = _XpressExpr( x_const, x_linear_vars, x_linear_coefs,
                                    x_quad_var_1, x_quad_var_2, x_quad_coefs )
        else:
            new_expr = _XpressExpr( x_const, x_linear_vars, x_linear_coefs )

        return new_expr, referenced_vars

    def _get_expr_from_pyomo_expr(self, expr, max_degree=2, objective=False):
        if max_degree == 2:
            repn = generate_standard_repn(expr, quadratic=True)
        else:
            repn = generate_standard_repn(expr, quadratic=False)

        try:
            xpress_expr, referenced_vars = self._get_expr_from_pyomo_repn(repn, max_degree, objective)
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {0}'.format(expr)
            raise DegreeError(msg)

        return xpress_expr, referenced_vars

    def _add_var(self, var, var_data=None):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vartype = self._xpress_vartype_from_var(var)
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = -self._xpress.infinity
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = self._xpress.infinity
        if var.is_fixed():
            lb = value(var.value)
            ub = value(var.value)

        #'''
        xp_var_data = _VariableData() if var_data is None else var_data
        xp_var_data.add(lb=lb, ub=ub, vartype=vartype, name=varname, idx=self._var_idx_count)

        if var_data is None:
            xp_var_data.store_in_xpress(self._solver_model)

        '''
        self._solver_model.addcols(objx=[0.], mstart=[0,0], mrwind=[], dmatval=[], 
                                    bdl=[lb], bdu=[ub], names=[varname], types=[vartype])
        if vartype == 'B' and (lb > 0 or ub < 1):
            if lb == ub:
                chgbounds_vals = [lb]
                chgbounds_type = ['B']
                chgbounds_idx = [self._var_idx_count]
            else:
                chgbounds_vals = [lb, ub]
                chgbounds_type = ['L', 'U']
                chgbounds_idx = [self._var_idx_count, self._var_idx_count]

            self._solver_model.chgbounds(chgbounds_idx, chgbounds_type, chgbounds_vals)
        '''

        self._pyomo_var_to_solver_var_map[var] = varname 
        self._solver_var_to_pyomo_var_map[varname] = var
        self._pyomo_var_to_var_idx_map[var] = self._var_idx_count
        self._var_idx_count += 1
        self._referenced_variables[var] = 0

    def _set_instance(self, model, kwds={}):
        self._range_constraints = dict()
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()

        self._pyomo_var_to_var_idx_map = ComponentMap()
        self._var_idx_count = 0
        self._con_idx_count = 0
        self._pyomo_sos_to_sos_idx_map = dict()
        self._sos_idx_count = 0

        try:
            if model.name is not None:
                self._solver_model = self._xpress.problem(name=model.name)
            else:
                self._solver_model = self._xpress.problem()
        except Exception:
            e = sys.exc_info()[1]
            msg = ("Unable to create Xpress model. "
                   "Have you installed the Python "
                   "bindings for Xpress?\n\n\t" +
                   "Error message: {0}".format(e))
            raise Exception(msg)
        self._add_block(model)

    def _add_block(self, block):
        var_data = _VariableData()
        for var in block.component_data_objects(
                ctype=pyomo.core.base.var.Var,
                descend_into=True,
                active=True,
                sort=True):
            self._add_var(var, var_data)
        var_data.store_in_xpress(self._solver_model)

        con_data = _ConstraintData(self._solver_model)
        for sub_block in block.block_data_objects(descend_into=True,
                                                  active=True):
            for con in sub_block.component_data_objects(
                    ctype=pyomo.core.base.constraint.Constraint,
                    descend_into=False,
                    active=True,
                    sort=True):
                if (not con.has_lb()) and \
                   (not con.has_ub()):
                    assert not con.equality
                    continue  # non-binding, so skip
                self._add_constraint(con, con_data)

            for con in sub_block.component_data_objects(
                    ctype=pyomo.core.base.sos.SOSConstraint,
                    descend_into=False,
                    active=True,
                    sort=True):
                self._add_sos_constraint(con)

            obj_counter = 0
            for obj in sub_block.component_data_objects(
                    ctype=pyomo.core.base.objective.Objective,
                    descend_into=False,
                    active=True):
                obj_counter += 1
                if obj_counter > 1:
                    raise ValueError("Solver interface does not "
                                     "support multiple objectives.")
                self._set_objective(obj)
        con_data.store_in_xpress()

    def _add_constraint(self, con, con_data=None):
        if not con.active:
            return None

        if is_fixed(con.body):
            if self._skip_trivial_constraints:
                return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        if con._linear_canonical_form:
            xpress_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                con.canonical_form(),
                self._max_constraint_degree)
        else:
            xpress_expr, referenced_vars = self._get_expr_from_pyomo_expr(
                con.body,
                self._max_constraint_degree) 

        if con.has_lb():
            if not is_fixed(con.lower):
                raise ValueError("Lower bound of constraint {0} "
                                 "is not constant.".format(con))
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError("Upper bound of constraint {0} "
                                 "is not constant.".format(con))

        if con.equality:
            x_sense = 'E'
            x_rhs = value(con.lower) - xpress_expr.constant
            x_range = 0.

        elif con.has_lb() and con.has_ub():
            x_sense = 'R'
            lb = value(con.lower)
            ub = value(con.upper)
            x_rhs = ub - xpress_expr.constant
            x_range = ub - lb

            self._range_constraints[conname] = x_range
            #x_range = [x_range]

        elif con.has_lb():
            x_sense = 'G'
            x_rhs = value(con.lower) - xpress_expr.constant
            x_range = 0.

        elif con.has_ub():
            x_sense = 'L'
            x_rhs = value(con.upper) - xpress_expr.constant
            x_range = 0.

        else:
            raise ValueError("Constraint does not have a lower "
                             "or an upper bound: {0} \n".format(con))

        #'''
        xpress_con_data = _ConstraintData(self._solver_model) if con_data is None else con_data

        if xpress_expr.quad_coefs is None:
            xpress_con_data.add(x_sense, x_rhs, xpress_expr.linear_vars, xpress_expr.linear_coefs,
                                conname, x_range)

        else:
            if x_sense in ['R', 'E']:
                raise RuntimeError("Xpress does not support range or equality quadratic constraints.")
            xpress_con_data.add(x_sense, x_rhs, xpress_expr.linear_vars, xpress_expr.linear_coefs,
                                conname, x_range, self._con_idx_count, xpress_expr.quad_vars_1,
                                xpress_expr.quad_vars_2, xpress_expr.quad_coefs)

        if con_data is None:
            xpress_con_data.store_in_xpress()
        '''
        self._solver_model.addrows([x_sense], [x_rhs], [0,len(xpress_expr.linear_vars)], 
                                    xpress_expr.linear_vars, xpress_expr.linear_coefs,
                                    range=x_range, names=[conname])

        if xpress_expr.quad_coefs is not None:
            if x_sense in ['R', 'E']:
                raise RuntimeError("Xpress does not support range or equality quadratic constraints.")
            self._solver_model.addqmatrix(self._con_idx_count, xpress_expr.quad_vars_1, xpress_expr.quad_vars_2, xpress_expr.quad_coefs)
        #'''

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_map[con] = conname 
        self._solver_con_to_pyomo_con_map[conname] = con

        #self._pyomo_con_to_con_idx_map[con] = self._con_idx_count
        self._con_idx_count += 1

    def _add_sos_constraint(self, con):
        if not con.active:
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)
        level = con.level
        if level not in [1,2]:
            raise ValueError("Solver does not support SOS "
                             "level {0} constraints".format(level))

        xpress_vars = list()
        weights = list()

        self._vars_referenced_by_con[con] = ComponentSet()

        if hasattr(con, 'get_items'):
            # aml sos constraint
            sos_items = list(con.get_items())
        else:
            # kernel sos constraint
            sos_items = list(con.items())

        for v, w in sos_items:
            self._vars_referenced_by_con[con].add(v)
            xpress_vars.append(self._pyomo_var_to_var_idx_map[v])
            self._referenced_variables[v] += 1
            weights.append(w)

        xpress_con = self._xpress.sos(xpress_vars, weights, level, conname)
        self._solver_model.addSOS(xpress_con)
        self._pyomo_con_to_solver_con_map[con] = conname 
        self._solver_con_to_pyomo_con_map[conname] = con

        self._pyomo_sos_to_sos_idx_map[con] = self._sos_idx_count
        self._sos_idx_count += 1

    def _xpress_vartype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate xpress variable type
        :param var: pyomo.core.base.var.Var
        :return: xpress.continuous or xpress.binary or xpress.integer
        """
        if var.is_binary():
            vartype = 'B'
        elif var.is_integer():
            vartype = 'I'
        elif var.is_continuous():
            vartype = 'C'
        else:
            raise ValueError('Variable domain type is not recognized for {0}'.format(var.domain))
        return vartype
    
    def _set_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[var] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == minimize:
            sense = self._xpress.minimize
        elif obj.sense == maximize:
            sense = self._xpress.maximize
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        xpress_expr, referenced_vars = self._get_expr_from_pyomo_expr(obj.expr, self._max_obj_degree,
                                                                        objective=True)

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        # this resets the objective
        self._solver_model.setObjective(xpress_expr.constant, sense=sense)
        self._solver_model.chgobj(xpress_expr.linear_vars, xpress_expr.linear_coefs)
        if xpress_expr.quad_coefs is not None:
            self._solver_model.chgmqobj(xpress_expr.quad_vars_1, xpress_expr.quad_vars_2, \
                                        xpress_expr.quad_coefs)

        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

    def _postsolve(self):
        # the only suffixes that we extract from XPRESS are
        # constraint duals, constraint slacks, and variable
        # reduced-costs. scan through the solver suffix list
        # and throw an exception if the user has specified
        # any others.
        extract_duals = False
        extract_slacks = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, "dual"):
                extract_duals = True
                flag = True
            if re.match(suffix, "slack"):
                extract_slacks = True
                flag = True
            if re.match(suffix, "rc"):
                extract_reduced_costs = True
                flag = True
            if not flag:
                raise RuntimeError("***The xpress_direct solver plugin cannot extract solution suffix="+suffix)

        xprob = self._solver_model
        xp = self._xpress
        xprob_attrs = xprob.attributes

        ## XPRESS's status codes depend on this
        ## (number of integer vars > 0) or (number of special order sets > 0)
        is_mip = (xprob_attrs.mipents > 0) or (xprob_attrs.sets > 0)

        if is_mip:
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
            extract_reduced_costs = False
            extract_duals = False

        self.results = SolverResults()
        soln = Solution()

        self.results.solver.name = self._name
        self.results.solver.wallclock_time = self._opt_time

        if is_mip:
            status = xprob_attrs.mipstatus
            mip_sols = xprob_attrs.mipsols
            if status == xp.mip_not_loaded:
                self.results.solver.status = SolverStatus.aborted
                self.results.solver.termination_message = "Model is not loaded; no solution information is available."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.unknown
            #no MIP solution, first LP did not solve, second LP did, third search started but incomplete
            elif status == xp.mip_lp_not_optimal \
                    or status == xp.mip_lp_optimal \
                    or status == xp.mip_no_sol_found:
                self.results.solver.status = SolverStatus.aborted
                self.results.solver.termination_message = "Model is loaded, but no solution information is available."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.unknown
            elif status == xp.mip_solution: # some solution available
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "Unable to satisfy optimality tolerances; a sub-optimal " \
                                                          "solution is available."
                self.results.solver.termination_condition = TerminationCondition.other
                soln.status = SolutionStatus.feasible
            elif status == xp.mip_infeas: # MIP proven infeasible
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "Model was proven to be infeasible"
                self.results.solver.termination_condition = TerminationCondition.infeasible
                soln.status = SolutionStatus.infeasible
            elif status == xp.mip_optimal: # optimal
                self.results.solver.status = SolverStatus.ok
                self.results.solver.termination_message = "Model was solved to optimality (subject to tolerances), " \
                                                          "and an optimal solution is available."
                self.results.solver.termination_condition = TerminationCondition.optimal
                soln.status = SolutionStatus.optimal
            elif status == xp.mip_unbounded and mip_sols > 0:
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "LP relaxation was proven to be unbounded, " \
                                                          "but a solution is available."
                self.results.solver.termination_condition = TerminationCondition.unbounded
                soln.status = SolutionStatus.unbounded
            elif status == xp.mip_unbounded and mip_sols <= 0:
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "LP relaxation was proven to be unbounded."
                self.results.solver.termination_condition = TerminationCondition.unbounded
                soln.status = SolutionStatus.unbounded
            else:
                self.results.solver.status = SolverStatus.error
                self.results.solver.termination_message = \
                    ("Unhandled Xpress solve status "
                     "("+str(status)+")")
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.error
        else: ## an LP, we'll check the lpstatus
            status = xprob_attrs.lpstatus
            if status == xp.lp_unstarted:
                self.results.solver.status = SolverStatus.aborted
                self.results.solver.termination_message = "Model is not loaded; no solution information is available."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.unknown
            elif status == xp.lp_optimal:
                self.results.solver.status = SolverStatus.ok
                self.results.solver.termination_message = "Model was solved to optimality (subject to tolerances), " \
                                                          "and an optimal solution is available."
                self.results.solver.termination_condition = TerminationCondition.optimal
                soln.status = SolutionStatus.optimal
            elif status == xp.lp_infeas:
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "Model was proven to be infeasible"
                self.results.solver.termination_condition = TerminationCondition.infeasible
                soln.status = SolutionStatus.infeasible
            elif status == xp.lp_cutoff:
                self.results.solver.status = SolverStatus.ok
                self.results.solver.termination_message = "Optimal objective for model was proven to be worse than the " \
                                                          "cutoff value specified; a solution is available."
                self.results.solver.termination_condition = TerminationCondition.minFunctionValue
                soln.status = SolutionStatus.optimal
            elif status == xp.lp_unfinished:
                self.results.solver.status = SolverStatus.aborted
                self.results.solver.termination_message = "Optimization was terminated by the user."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.error
            elif status == xp.lp_unbounded:
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "Model was proven to be unbounded."
                self.results.solver.termination_condition = TerminationCondition.unbounded
                soln.status = SolutionStatus.unbounded
            elif status == xp.lp_cutoff_in_dual:
                self.results.solver.status = SolverStatus.ok
                self.results.solver.termination_message = "Xpress reported the LP was cutoff in the dual."
                self.results.solver.termination_condition = TerminationCondition.minFunctionValue
                soln.status = SolutionStatus.optimal
            elif status == xp.lp_unsolved:
                self.results.solver.status = SolverStatus.error
                self.results.solver.termination_message = "Optimization was terminated due to unrecoverable numerical " \
                                                          "difficulties."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.error
            elif status == xp.lp_nonconvex:
                self.results.solver.status = SolverStatus.error
                self.results.solver.termination_message = "Optimization was terminated because nonconvex quadratic data " \
                                                          "were found."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.error
            else:
                self.results.solver.status = SolverStatus.error
                self.results.solver.termination_message = \
                    ("Unhandled Xpress solve status "
                     "("+str(status)+")")
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.error


        self.results.problem.name = xprob_attrs.matrixname

        if xprob_attrs.objsense == 1.0:
            self.results.problem.sense = minimize
        elif xprob_attrs.objsense == -1.0:
            self.results.problem.sense = maximize
        else:
            raise RuntimeError('Unrecognized Xpress objective sense: {0}'.format(xprob_attrs.objsense))

        self.results.problem.upper_bound = None
        self.results.problem.lower_bound = None
        if not is_mip: #LP or continuous problem
            try:
                self.results.problem.upper_bound = xprob_attrs.lpobjval
                self.results.problem.lower_bound = xprob_attrs.lpobjval
            except (self._XpressException, AttributeError):
                pass
        elif xprob_attrs.objsense == 1.0:  # minimizing MIP
            try:
                self.results.problem.upper_bound = xprob_attrs.mipbestobjval
            except (self._XpressException, AttributeError):
                pass
            try:
                self.results.problem.lower_bound = xprob_attrs.bestbound
            except (self._XpressException, AttributeError):
                pass
        elif xprob_attrs.objsense == -1.0:  # maximizing MIP
            try:
                self.results.problem.upper_bound = xprob_attrs.bestbound
            except (self._XpressException, AttributeError):
                pass
            try:
                self.results.problem.lower_bound = xprob_attrs.mipbestobjval
            except (self._XpressException, AttributeError):
                pass
        else:
            raise RuntimeError('Unrecognized xpress objective sense: {0}'.format(xprob_attrs.objsense))

        try:
            soln.gap = self.results.problem.upper_bound - self.results.problem.lower_bound
        except TypeError:
            soln.gap = None

        self.results.problem.number_of_constraints = xprob_attrs.rows + xprob_attrs.sets + xprob_attrs.qconstraints
        self.results.problem.number_of_nonzeros = xprob_attrs.elems
        self.results.problem.number_of_variables = xprob_attrs.cols
        self.results.problem.number_of_integer_variables = xprob_attrs.mipents
        self.results.problem.number_of_continuous_variables = xprob_attrs.cols - xprob_attrs.mipents
        self.results.problem.number_of_objectives = 1
        self.results.problem.number_of_solutions = xprob_attrs.mipsols if is_mip else 1 

        # if a solve was stopped by a limit, we still need to check to
        # see if there is a solution available - this may not always
        # be the case, both in LP and MIP contexts.
        if self._save_results:
            """
            This code in this if statement is only needed for backwards compatability. It is more efficient to set
            _save_results to False and use load_vars, load_duals, etc.
            """
            if xprob_attrs.lpstatus in \
                    [xp.lp_optimal, xp.lp_cutoff, xp.lp_cutoff_in_dual] or \
                    xprob_attrs.mipsols > 0:
                soln_variables = soln.variable
                soln_constraints = soln.constraint

                var_vals = xprob.getSolution()
                var_names = xprob.getnamelist(2)
                for xpress_var, val in zip(var_names, var_vals):
                    pyomo_var = self._solver_var_to_pyomo_var_map[xpress_var]
                    if self._referenced_variables[pyomo_var] > 0:
                        pyomo_var.stale = False
                        soln_variables[xpress_var] = {"Value": val}

                if extract_reduced_costs:
                    rc_vals = xprob.getRCost()
                    for xpress_var, val in zip(var_names, rc_vals):
                        pyomo_var = self._solver_var_to_pyomo_var_map[xpress_var]
                        if self._referenced_variables[pyomo_var] > 0:
                            soln_variables[xpress_var]["Rc"] = val

                if extract_duals or extract_slacks:
                    if self._pyomo_con_to_solver_con_map:
                        con_names = xprob.getnamelist(1)
                    else:
                        con_names = list()
                    for con in con_names:
                        soln_constraints[con] = {}

                if extract_duals:
                    vals = xprob.getDual()
                    for con, val in zip(con_names, vals):
                        soln_constraints[con]["Dual"] = val

                if extract_slacks:
                    vals = xprob.getSlack()
                    range_constraints = self._range_constraints
                    for con, val in zip(con_names, vals):
                        if con in range_constraints:
                            ## for xpress, the slack on a range constraint
                            ## is based on the upper bound
                            x_range = range_constraints[con]
                            ub_s = val
                            lb_s = val - x_range
                            if abs(ub_s) > abs(lb_s):
                                soln_constraints[con]["Slack"] = ub_s
                            else:
                                soln_constraints[con]["Slack"] = lb_s
                        else:
                            soln_constraints[con]["Slack"] = val

        elif self._load_solutions:
            if xprob_attrs.lpstatus == xp.lp_optimal and \
                    ((not is_mip) or (xprob_attrs.mipsols > 0)):

                self._load_vars()

                if extract_reduced_costs:
                    self._load_rc()

                if extract_duals:
                    self._load_duals()

                if extract_slacks:
                    self._load_slacks()

        self.results.solution.insert(soln)

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin.
        TempfileManager.pop(remove=not self._keepfiles)
        return DirectOrPersistentSolver._postsolve(self)

    def warm_start_capable(self):
        return True

    def _warm_start(self):
        mipsolval = list()
        mipsolcol = list()
        for pyomo_var, xpress_var in self._pyomo_var_to_var_idx_map.items():
            if pyomo_var.value is not None:
                mipsolval.append(value(pyomo_var))
                mipsolcol.append(xpress_var)
        self._solver_model.addmipsol(mipsolval, mipsolcol)

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_var_idx_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        xpress_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getSolution(xpress_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                var.stale = False
                var.value = val

    def _load_rc(self, vars_to_load=None):
        if not hasattr(self._pyomo_model, 'rc'):
            self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)
        var_map = self._pyomo_var_to_var_idx_map
        ref_vars = self._referenced_variables
        rc = self._pyomo_model.rc
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        xpress_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getRCost(xpress_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                rc[var] = val

    def _load_duals(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        dual = self._pyomo_model.dual

        if cons_to_load is None:
            cons_to_load = con_map.keys()
        xpress_cons_to_load = [con_map[pyomo_con] for pyomo_con in cons_to_load]
        vals = self._solver_model.getDual(xpress_cons_to_load)

        for pyomo_con, val in zip(cons_to_load, vals):
            dual[pyomo_con] = val

    def _load_slacks(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'slack'):
            self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        slack = self._pyomo_model.slack

        if cons_to_load is None:
            cons_to_load = con_map.keys()
        xpress_cons_to_load = [con_map[pyomo_con] for pyomo_con in cons_to_load]
        vals = self._solver_model.getSlack(xpress_cons_to_load)
        range_constraints = self._range_constraints

        for pyomo_con, xpress_con, val in zip(cons_to_load, xpress_cons_to_load, vals):
            if xpress_con in range_constraints:
                ## for xpress, the slack on a range constraint
                ## is based on the upper bound
                x_range = range_constraints[con]
                ub_s = val
                lb_s = val - x_range
                if abs(ub_s) > abs(lb_s):
                    slack[pyomo_con] = ub_s
                else:
                    slack[pyomo_con] = lb_s
            else:
                slack[pyomo_con] = val

    def load_duals(self, cons_to_load=None):
        """
        Load the duals into the 'dual' suffix. The 'dual' suffix must live on the parent model.

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        self._load_duals(cons_to_load)

    def load_rc(self, vars_to_load=None):
        """
        Load the reduced costs into the 'rc' suffix. The 'rc' suffix must live on the parent model.

        Parameters
        ----------
        vars_to_load: list of Var
        """
        self._load_rc(vars_to_load)

    def load_slacks(self, cons_to_load=None):
        """
        Load the values of the slack variables into the 'slack' suffix. The 'slack' suffix must live on the parent
        model.

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        self._load_slacks(cons_to_load)
