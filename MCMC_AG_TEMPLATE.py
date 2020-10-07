#emcee analysis for GRB afterglow
#
#Gavin P Lamb 01/10/2018 gpl6@le.ac.uk
from multiprocessing import Pool
import tqdm
import numpy as np
import corner
import emcee
from os.path import join
from astropy.table import Table, Column

import joelib.physics.jethead_expansion as Exp
from joelib.constants.constants import *



""
#I'VE LEFT THE DATA IN HERE, YOU CAN USE THIS TO TEST AGAINST YOUR OUTPUT - NOTE THAT THERE ARE FOUR(4) FREQUENCIES - BUT I'M SURE YOU CAN FIGURE IT OUT.
#ALTERNATIVELY, JUST USE THE DATA YOU HAVE HOWEVER IS EASIEST ;-)

#Updated optical and X-ray data from Fong ea 2019, and Hajela ea 2019
time = np.array([16.42, 17.39, 18.33, 22.36, 31.22, 31.32, 46.26, 54.27, 57.22, 93.13, 115., 163., 197., 216.91, 220, 256.76, 267, 272.67, 288.61, 294, 80.1, 112.04, 125.3, 162.89, 162.89, 149.26, 181.64, 216.88, 272.61, 288.55, 110.49, 137.04, 165.19, 172.21, 209.10, 218.37, 296.80, 328.22, 362.32, 9.2, 15.64, 109.52, 158.64, 259.99, 358.61, 582.30, 741.72])
# time in days

data = np.array([0.0187, 0.0151, 0.0145, 0.0225, 0.034, 0.034, 0.044, 0.048, 0.061, 0.07, 0.089, 0.09785, 0.07895, 0.0647, 0.069, 0.055, 0.0403, 0.044, 0.035, 0.0312, 0.0374, 0.0577, 0.082, 0.0808, 0.0611, 0.0989, 0.0896, 0.039, 0.036, 0.035, 0.11e-3, 0.084e-3, 0.091e-3, 0.085e-3, 0.082e-3, 0.063e-3, 0.044e-3, 0.034e-3, 0.027e-3, 2.52e-7, 6.69e-7, 20.03e-7, 21.7e-7, 10.82e-7, 7.82e-7, 2.17e-7, 1.39e-7])
# flux in mJy

freq = np.array([3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 3.e9, 6e9, 6e9, 6e9, 6e9, 6e9, 6e9, 6e9, 6e9, 6e9, 6e9, 5.09e14, 5.09e14, 5.09e14, 5.09e14, 5.09e14, 5.09e14, 5.09e14, 5.09e14, 5.09e14, 2.418e17 ,2.418e17, 2.418e17 ,2.418e17 ,2.418e17, 2.418e17, 2.418e17, 2.418e17])
# frequency of obs.

err = np.array([0.0063, 0.0039, 0.0037, 0.0029, 0.0036, 0.0036, 0.0040, 0.006, 0.009, 0.0057, 0.0246, 0.0279, 0.0107, 0.0027, 0.015, 0.0011, 0.0027, 0.007, 0.007, 0.0036, 0.0042, 0.0047, 0.0093, 0.0125, 0.0073, 0.0085, 0.0133, 0.009, 0.007, 0.007, 0.019e-3, 0.018e-3, 0.016e-3, 0.017e-3, 0.02e-3, 0.018e-3, 0.014e-3, 0.011e-3, 0.0072e-3, 1.18e-7, 1.78e-7, 2.4e-7, 3.23e-7, 2.55e-7, 2.72e-7, 0.69e-7, 0.54e-7])
# error on flux

freqs = np.array([3.e9, 6.e9, 5.09e14, 2.481e17])

#The model afterglow --- Put your afterglow scipt here


def lnlike(prior, time, data, freq, err):
	E1, G1, thc1, incl1, EB1, Ee1, n1, p = prior
	incl1 = np.arccos(incl1)
	#model parameters

	EB, Ee, pp, nn, Eps0, G0 = 10.**EB1, 10.**Ee1, p, 10.**n1, 10.**E1, G1
	#microphysical magnetic EB, microphysical electric Ee, particle index p
	#ambient number density n, isotropic kinetic energy Eiso, Lorentz factor Gc

	#THESE ^ ARE THE MCMC PRIOR PARAMETERS
	#USE THESE TO FEED AN ITERATION OF YOUR CODE

	#PUT YOUR FIXED PARAMETERS HERE (I FIX THETA_J) AND CALL YOUR SCRIPT


	jet = Exp.jetHeadGauss(Eps0, G0, nn, Ee, EB, pp, 1000, 1e13, 1e22, "peer", 100, 20.*pi/180., thc1, aa=1, withSpread=False)

	LC= array([])

	for frequency in freqs:
		file = [freq=frequency]
		times = time[fil]

		tt, lc, _, _ = Exp.light_curve_peer_SJ(jet, pp, incl1, frequency, 41.3e6*pc, "discrete", times, 1)
		LC = concatenate([LC, lc])


	#RETURN AN ARRAY OF FLUX VALUES AT TIMES AND FREQUENCIES THAT ARE THE SAME AS THE OBSERVATIONS
	#I CALLED THIS flx AND IT IS IN THE SAME UNITS AS THE data AND err ARRAYS ABOVE
	flx = LC*1.e26 #erg/s/cm^2/Hz to mJy
	#THIS IS THE LIKLIHOOD FUNCTION - DON'T OVER THINK IT, WE JUST NEED TO MINIMISE THE ERROR, NOTHING FANCY -- IT USES THE data AND err WITH THE SCRIPT flx VALUES
	like = -0.5*(np.sum(((data-flx)/err)**2.))#+np.log(err**2)
	return like

def lnprior(prior):
	E1, E2, G1, G2, thc1, incl1, EB1, Ee1, n1, p = prior
	#THESE ARE THE PRIOR LIMITS -- NOTE THAT SOME OF THESE ARE LOGARITHMIC (see line 37 above, or in cosine space)
	if 47. <= E1 <= 54. and 2. <= G1 <= 1000. and 0.0175 <= thc1 <= 0.4363 and 0.9004 <= incl1 <= 0.9689 and -5.<= EB1 <= -0.5 and -5. <= Ee1 <= -0.5 and -6. <= n1 <= 0. and 2.01 <= p <= 2.99:
		return 0
	return -np.inf

def lnprob(prior, time, data, freq, err):
	lp = lnprior(prior)
	if np.isinf(lp):
		return -np.inf
	return lp+lnlike(prior, time, data, freq, err)



if __name__ == "__main__":
	nwalkers = 50  # minimum is 2*ndim, more the merrier
	ndim = 8 #THIS IS THE NUMBER OF FREE PARAMETERS THAT YOU ARE FITTING
	burnsteps = 1000
	nsteps = 10000  # MCMC steps, should be >1e3
	thin = 1 #THIS REMOVES SOME OF THE SAMPLE ie thin = 2 WOULD ONLY SAMPLE EVERY SECOND POSTERIOR (GOOD FOR VERY LARGE DATA SETS)
	p0 = np.zeros([nwalkers,ndim],float)
	# Initial parameter guess
	print ("Initial parameter guesses")
	p0[:, 0] = 51 + np.random.randn(nwalkers)*0.1  # sample E1
	p0[:, 1] = 100 + np.random.randn(nwalkers)*10  # sample G1
	p0[:, 3] = 0.12 + np.random.randn(nwalkers)*0.01  # sample thc1
	p0[:, 4] = 0.91 + np.random.randn(nwalkers)*0.01  # sample cos(incl1)
	p0[:, 5] = -2.1 + np.random.randn(nwalkers)*0.1  # sample EB1
	p0[:, 6] = -2.1 + np.random.randn(nwalkers)*0.1  # sample Ee1
	p0[:, 7] = -3.1 + np.random.randn(nwalkers)*0.1  # sample n1
	p0[:, 8] = 2.16 + np.random.randn(nwalkers)*0.01  # sample p
	# Multiprocess
	filename = "backend-refreshed-AG.h5" #THIS IS A SPECIAL FILE THAT MEANS YOU CAN (I CAN SHOW YOU) RESTART A RUN WHERE YOU LEFT OFF
	backend = emcee.backends.HDFBackend(filename)
	backend.reset(nwalkers, ndim)

	pool = Pool()
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, data, freq, err), pool=pool, backend=backend)
	""
	# Burn in run
	print ("Burning in")
	pos, prob, state = sampler.run_mcmc(p0, burnsteps, progress=True)
	sampler.reset()
	""
	# Production run
	print ("Production run")
	sampler.run_mcmc(pos, nsteps, progress=True)
