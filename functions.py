import cgp
import numpy as np


class sat_add(cgp.OperatorNode):
  """
  Saturated addition cgp function
  """
  _arity = 2
  _def_output = "np.clip(np.add(x_0, x_1), -4.0, 4.0)"
  _def_numpy_output = "np.clip(np.add(x_0, x_1), -4.0, 4.0)"


class sat_sub(cgp.OperatorNode):
  """
  Saturated subtraction cgp function
  """
  _arity = 2
  _def_output = "np.clip(np.subtract(x_0, x_1), -4.0, 4.0)"
  _def_numpy_output = "np.clip(np.subtract(x_0, x_1), -4.0, 4.0)"


class cgp_min(cgp.OperatorNode):
  """
  Min cgp function
  """
  _arity = 2
  _def_output = "np.min((x_0, x_1), axis=0)"
  _def_numpy_output = "np.min((x_0, x_1), axis=0)"


class cgp_max(cgp.OperatorNode):
  """
  Max cgp function
  """
  _arity = 2
  _def_output = "np.max((x_0, x_1), axis=0)"
  _def_numpy_output = "np.max((x_0, x_1), axis=0)"


class greater_than(cgp.OperatorNode):
  """
  Greater than cgp function (returns 1 if x_0 > x_1, else -1)
  """
  _arity = 2
  _def_output = "np.where(x_0 > x_1, 1.0, -1.0)"
  _def_numpy_output = "np.where(x_0 > x_1, 1.0, -1.0)"


class sat_mul(cgp.OperatorNode):
  """
  Saturated multiplication cgp function
  """
  _arity = 2
  _def_output = "np.clip(x_0*x_1, -8.0, 8.0)"
  _def_numpy_output = "np.clip(x_0*x_1, -8.0, 8.0)"


class const_random(cgp.ConstantFloat):
  """
  Constant float from 0 to 1
  """
  _arity = 0
  _initial_values = {"<w>": lambda: np.random.rand()}
  _def_output = "<w>"
  _def_numpy_output = "<w>"


class scale_up(cgp.OperatorNode):
  """
  CGP functions which dooubles the input
  """
  _arity = 1
  _def_output = "np.multiply(x_0, 2)"
  _def_numpy_output = "np.multiply(x_0, 2)"


class scale_down(cgp.OperatorNode):
  """
  CGP functions which halves the input
  """
  _arity = 1
  _def_output = "np.divide(x_0, 2)"
  _def_numpy_output = "np.divide(x_0, 2)"


class negation(cgp.OperatorNode):
  """
  CGP functions which negates the input
  """
  _arity = 1
  _def_output = "np.negative(x_0)"
  _def_numpy_output = "np.negative(x_0)"


class continous_and(cgp.OperatorNode):
  """
  CGP functions which performs a 'continous and' operation on the input
  If both x1 and x2 are positive -> return bigger number, else return the negative number
  """
  _arity = 2
  _def_output = "np.where(np.logical_and(x_0 > 0, x_1 > 0), np.max((x_0, x_1), axis=0), np.min((x_0, x_1), axis=0))"
  _def_numpy_output = "np.where(np.logical_and(x_0 > 0, x_1 > 0), np.max((x_0, x_1), axis=0), np.min((x_0, x_1), axis=0))"


class continous_or(cgp.OperatorNode):
  """
  CGP functions which performs a 'continous or' operation on the input
  If both x1 and x2 are negative -> return smaller number, else return the positive number
  """
  _arity = 2
  _def_output = "np.where(np.logical_and(x_0 < 0, x_1 < 0), np.min((x_0, x_1), axis=0), np.max((x_0, x_1), axis=0))"
  _def_numpy_output = "np.where(np.logical_and(x_0 < 0, x_1 < 0), np.min((x_0, x_1), axis=0), np.max((x_0, x_1), axis=0))"


class const_float(cgp.ConstantFloat):
  _def_output = "5"
  _def_numpy_output = "5"


class multiplexor(cgp.OperatorNode):
  """A node that outputs the value of its second input if its first input
  is non-negative, and the value of its third input otherwise."""

  _arity = 3
  _def_output = "x_1 if x_0 > 0 else x_2"
  _def_numpy_output = "np.piecewise(x_0, [x_0 >= 0, x_0 < 0], [x_1[x_0 >= 0] , x_2[x_0 < 0]])"
