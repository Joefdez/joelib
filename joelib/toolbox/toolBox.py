from numpy import *
from numpy.random import uniform, seed
from matplotlib.pylab import *
from scipy.integrate import quad




def quadArray(func, aa, bb):
     """
     Wrapper to use quad function over an array of arguments for the same function
     func :function to be integrated
     aa, bb: arrays containing integration limits. Must be 1-D and of the same length
     """
     num     = shape(aa)[0]
     results = zeros(num)

     for ii in range(num):
         results[ii] = quad(func, aa[ii], bb[ii])[0]

     return results


def uniformLog(aa, bb, nn):
    """
    Return a nn random numbers uniformly distributed in log10 space
    """
    seed()
    limA = log10(aa)
    limB = log10(bb)

    nums = uniform(limA, limB, nn)

    return 10.**nums


#def uniformLogVarlim(aa, bb):
#    seed()
#    limA = log10(aa)
#    limB = log10(bb)


def rk4(dydr, obj, x0, y0, hh ):


      "Apply Runge Kutta Formulas to find next value of y"
      k1 = hh * dydr(obj, y0, x0)
      k2 = hh * dydr(obj, y0 + 0.5 * k1, x0 + 0.5 * hh)
      k3 = hh * dydr(obj, y0 + 0.5 * k2, x0 + 0.5 * hh)
      k4 = hh * dydr(obj, y0 + k3, x0 + hh)

      # Update next value of y
      yy = y0 + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)


      return yy

def rk4_latexp2(dydr, x0, y0, hh, params):


      "Apply Runge Kutta Formulas to find next value of y"
      k1 = hh * dydr(y0, x0)
      k2 = hh * dydr(y0 + 0.5 * k1, x0 + 0.5 * hh)
      k3 = hh * dydr(y0 + 0.5 * k2, x0 + 0.5 * hh)
      k4 = hh * dydr(y0 + k3, x0 + hh)

      # Update next value of y
      yy = y0 + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)


      return yy




def weighted_average_std(values, weights, keepdims=False):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    av = average(values, weights=weights)
    # Fast and numerically precise:
    var = average((values-av)**2, weights=weights)
    return (av, sqrt(var))
