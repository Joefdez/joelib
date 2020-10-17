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




#I'VE LEFT THE DATA IN HERE, YOU CAN USE THIS TO TEST AGAINST YOUR OUTPUT - NOTE THAT THERE ARE FOUR(4) FREQUENCIES - BUT I'M SURE YOU CAN FIGURE IT OUT.
#ALTERNATIVELY, JUST USE THE DATA YOU HAVE HOWEVER IS EASIEST ;-)



time = np.array([  9.2,  14.9,  16.4,  17.4,  18.3,  18.7,  19.4,  21.4,  22.4,
        23.4,  24.2,  31.3,  35.3,  39.2,  46.3,  53.3,  54.3,  57.2,
        65.9,  66.6,  67.2,  72.2,  75.5,  75.5,  77.6,  79.2,  80.1,
        92.4,  93.1,  93.1,  93.2,  97.1, 107. , 107. , 109. , 109. ,
       111. , 112. , 115. , 115. , 115. , 125. , 125. , 126. , 133. ,
       137. , 149. , 150. , 152. , 158. , 161. , 163. , 163. , 163. ,
       163. , 165. , 167. , 170. , 172. , 183. , 197. , 197. , 207. ,
       209. , 216. , 217. , 217. , 217. , 217. , 218. , 218. , 222. ,
       229. , 252. , 257. , 259. , 261. , 267. , 267. , 273. , 273. ,
       289. , 289. , 294. , 297. , 298. , 320. , 324. , 328. , 357. ,
       359. , 362. , 380. , 489. , 545. , 580. , 581. , 741. , 767. ,
       938. ])



data = np.array([5.66e-04, 6.58e-04, 1.87e+01, 1.51e+01, 1.45e+01, 1.54e+01, 1.59e+01, 1.36e+01, 2.25e+01, 2.00e+01, 2.56e+01, 3.40e+01, 4.40e+01, 2.28e+01, 4.40e+01, 3.20e+01, 4.80e+01, 6.10e+01,
       1.48e+02, 9.80e+01, 4.26e+01, 5.80e+01, 3.59e+01, 3.96e+01, 7.70e+01, 4.50e+01, 4.17e+01, 3.17e+01, 9.80e+01, 7.00e+01, 2.60e+01, 1.99e+02, 1.27e+02, 5.32e+01, 2.96e-03, 1.09e-01,
       1.11e-01, 6.29e+01, 9.62e+01, 5.12e+01, 4.12e+01, 5.82e+01, 1.28e+02, 2.21e+02, 3.37e-03, 8.40e-02, 6.06e+01, 9.00e+01, 1.84e+02, 3.03e-03, 2.66e-03, 9.73e+01, 6.73e+01, 4.74e+01,
       3.96e+01, 9.10e-02, 5.79e+01, 1.13e-01, 8.50e-02, 2.11e+02, 7.59e+01, 8.93e+01, 4.20e+01, 8.20e-02, 3.63e+01, 6.05e+01, 4.17e+01, 3.26e+01, 2.47e+01, 6.47e+01, 6.30e-02, 3.97e+01,
       4.80e+01, 7.13e+01, 4.32e+01, 1.55e-03, 6.26e+01, 2.50e+01, 4.03e+01, 3.48e+01, 2.72e+01, 3.63e+01, 2.70e+01, 3.12e+01, 4.40e-02, 2.34e+01, 2.31e+01, 4.72e+01, 3.40e-02, 9.70e-04,
       1.55e+01, 2.70e-02, 3.79e+01, 1.48e+01, 5.90e+00, 1.80e+01, 3.54e-04, 2.68e-04, 4.90e+00, 1.95e-04])

freq = np.array([2.41e+17, 2.41e+17, 3.00e+09, 3.00e+09, 3.00e+09, 7.25e+09,
       6.20e+09, 6.20e+09, 3.00e+09, 6.00e+09, 3.00e+09, 3.00e+09,
       1.50e+09, 6.00e+09, 3.00e+09, 6.00e+09, 3.00e+09, 3.00e+09,
       6.70e+08, 1.30e+09, 6.00e+09, 4.50e+09, 7.35e+09, 7.35e+09,
       1.40e+09, 4.50e+09, 6.00e+09, 7.25e+09, 1.50e+09, 3.00e+09,
       1.50e+10, 6.70e+08, 1.30e+09, 1.30e+09, 2.41e+17, 3.80e+14,
       5.06e+14, 6.00e+09, 3.00e+09, 1.00e+10, 1.50e+10, 7.25e+09,
       1.30e+09, 6.70e+08, 2.41e+17, 5.06e+14, 7.25e+09, 5.10e+09,
       1.30e+09, 2.41e+17, 2.41e+17, 3.00e+09, 6.00e+09, 1.00e+10,
       1.50e+10, 5.06e+14, 7.25e+09, 3.80e+14, 5.06e+14, 6.50e+08,
       3.00e+09, 1.30e+09, 5.00e+09, 5.06e+14, 1.00e+10, 3.00e+09,
       6.00e+09, 1.00e+10, 1.50e+10, 3.00e+09, 5.06e+14, 7.25e+09,
       4.50e+09, 1.30e+09, 3.00e+09, 2.41e+17, 1.30e+09, 7.25e+09,
       3.00e+09, 3.00e+09, 6.00e+09, 3.00e+09, 6.00e+09, 3.00e+09,
       5.06e+14, 7.25e+09, 7.25e+09, 1.30e+09, 5.06e+14, 2.41e+17,
       7.25e+09, 5.06e+14, 1.30e+09, 3.00e+09, 6.00e+09, 7.25e+09,
       2.41e+17, 2.41e+17, 3.00e+09, 2.41e+17])


err = np.array([1.70e-04, 1.30e-04, 6.30e+00, 3.90e+00, 3.70e+00, 4.80e+00,
       5.50e+00, 2.90e+00, 3.40e+00, 3.10e+00, 2.90e+00, 3.60e+00,
       1.00e+01, 2.60e+00, 4.00e+00, 4.00e+00, 6.00e+00, 9.00e+00,
       2.20e+01, 2.00e+01, 4.10e+00, 5.00e+00, 4.30e+00, 7.00e+00,
       1.90e+01, 7.00e+00, 4.70e+00, 4.30e+00, 1.40e+01, 5.70e+00,
       4.40e+00, 1.60e+01, 1.80e+01, 4.50e+00, 2.60e-04, 1.70e-02,
       1.90e-02, 3.20e+00, 8.00e+00, 3.40e+00, 1.90e+00, 5.00e+00,
       2.10e+01, 1.90e+01, 4.00e-04, 1.80e-02, 4.30e+00, 3.00e+01,
       1.90e+01, 2.60e-04, 2.70e-04, 1.13e+01, 4.10e+00, 3.60e+00,
       2.00e+00, 1.60e-02, 6.90e+00, 1.90e-02, 1.70e-02, 3.40e+01,
       5.20e+00, 1.39e+01, 1.20e+01, 2.00e-02, 3.60e+00, 7.50e+00,
       7.50e+00, 4.00e+00, 3.10e+00, 2.70e+00, 1.80e-02, 7.20e+00,
       6.00e+00, 6.70e+00, 5.80e+00, 1.90e-04, 7.00e+00, 4.10e+00,
       2.70e+00, 4.90e+00, 2.10e+00, 3.90e+00, 2.80e+00, 3.60e+00,
       1.40e-02, 4.20e+00, 4.00e+00, 1.28e+01, 1.10e-02, 1.90e-04,
       5.00e+00, 7.00e-03, 1.18e+01, 2.90e+00, 1.90e+00, 4.20e+00,
       9.00e-05, 9.00e-05, 1.80e+00, 7.00e-05])


print(np.shape(time),np.shape(freq), np.shape(data), np.shape(err))
#freqs = np.array([3.00e9, 6.00e9, 2.41e17, 5.06e14])
#fil   = [(freq==3.00e9) | (freq==6.00e9) | (freq==2.41e17) | (freq==5.06e14)]
#time  = time[fil]
#data  = data[fil]
#freq  = freq[fil]
#err   = err[fil]

freqs = np.unique(freq)

data_ord = np.array([])
err_ord = np.array([])

for frequency in freqs:
	print(frequency)
	data_ord = np.concatenate([data_ord, data[freq==frequency]])
	err_ord = np.concatenate([err_ord, err[freq==frequency]])

print(np.shape(err_ord), np.shape(data_ord))


#The model afterglow --- Put your afterglow scipt here


def lnlike(prior, time, data_ord, freq, err_ord):
	E1, G1, thc1, incl1, EB1, Ee1, n1, p = prior
	incl1 = np.arccos(incl1)
	#model parameters

	EB, Ee, pp, nn, Eps0, G0 = 10.**EB1, 10.**Ee1, p, 10.**n1, 10.**E1, G1
	#microphysical magnetic EB, microphysical electric Ee, particle index p
	#ambient number density n, isotropic kinetic energy Eiso, Lorentz factor Gc

	#THESE ^ ARE THE MCMC PRIOR PARAMETERS
	#USE THESE TO FEED AN ITERATION OF YOUR CODE

	#PUT YOUR FIXED PARAMETERS HERE (I FIX THETA_J) AND CALL YOUR SCRIPT


	jet = Exp.jetHeadGauss(Eps0, G0, nn, Ee, EB, pp, 500, 1e14, 1e22, "peer", 50, 30.*pi/180., thc1, aa=1, withSpread=False)

	LC= np.array([])

	for frequency in freqs:
		times = time[freq==frequency]

		tt, lc, _, _ = Exp.light_curve_peer_SJ(jet, pp, incl1, frequency, 41.3e6*pc, "discrete", times, 1)
		LC = np.concatenate([LC, lc[0,:]])


	#RETURN AN ARRAY OF FLUX VALUES AT TIMES AND FREQUENCIES THAT ARE THE SAME AS THE OBSERVATIONS
	#I CALLED THIS flx AND IT IS IN THE SAME UNITS AS THE data AND err ARRAYS ABOVE
	flx = LC*1.e29 #erg/s/cm^2/Hz to mJy
	#THIS IS THE LIKLIHOOD FUNCTION - DON'T OVER THINK IT, WE JUST NEED TO MINIMISE THE ERROR, NOTHING FANCY -- IT USES THE data AND err WITH THE SCRIPT flx VALUES
	like = -0.5*(np.sum(((data_ord-flx)/err_ord)**2.))#+np.log(err**2)
	return like

def lnprior(prior):
	E1, G1, thc1, incl1, EB1, Ee1, n1, p = prior
	#THESE ARE THE PRIOR LIMITS -- NOTE THAT SOME OF THESE ARE LOGARITHMIC (see line 37 above, or in cosine space)
	if 47. <= E1 <= 54. and 2. <= G1 <= 1000. and 0.0175 <= thc1 <= 0.4363 and 0.9004 <= incl1 <= 0.9689 and -5.<= EB1 <= -0.5 and -5. <= Ee1 <= -0.5 and -6. <= n1 <= 0. and 2.01 <= p <= 2.99:
		return 0
	return -np.inf

def lnprob(prior, time, data_ord, freq, err_ord):
	lp = lnprior(prior)
	if np.isinf(lp):
		return -np.inf
	return lp+lnlike(prior, time, data_ord, freq, err_ord)



if __name__ == "__main__":
	nwalkers = 32  # minimum is 2*ndim, more the merrier
	ndim = 8 #THIS IS THE NUMBER OF FREE PARAMETERS THAT YOU ARE FITTING
	burnsteps = 1000
	nsteps = 10000  # MCMC steps, should be >1e3
	thin = 1 #THIS REMOVES SOME OF THE SAMPLE ie thin = 2 WOULD ONLY SAMPLE EVERY SECOND POSTERIOR (GOOD FOR VERY LARGE DATA SETS)
	p0 = np.zeros([nwalkers,ndim],float)
	# Initial parameter guess
	print ("Initial parameter guesses")
	p0[:, 0] = 51 + np.random.randn(nwalkers)*0.1  # sample E1
	p0[:, 1] = 100 + np.random.randn(nwalkers)*10  # sample G1
	p0[:, 2] = 0.12 + np.random.randn(nwalkers)*0.01  # sample thc1
	p0[:, 3] = 0.91 + np.random.randn(nwalkers)*0.01  # sample cos(incl1)
	p0[:, 4] = -2.1 + np.random.randn(nwalkers)*0.1  # sample EB1
	p0[:, 5] = -2.1 + np.random.randn(nwalkers)*0.1  # sample Ee1
	p0[:, 6] = -3.1 + np.random.randn(nwalkers)*0.1  # sample n1
	p0[:, 7] = 2.16 + np.random.randn(nwalkers)*0.01  # sample p
	# Multiprocess
	filename = "backend-refreshed-AG_all.h5" #THIS IS A SPECIAL FILE THAT MEANS YOU CAN (I CAN SHOW YOU) RESTART A RUN WHERE YOU LEFT OFF
	backend = emcee.backends.HDFBackend(filename)
	backend.reset(nwalkers, ndim)

	pool = Pool()
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, data_ord, freq, err_ord), pool=pool, backend=backend)
	""
	# Burn in run
	print ("Burning in")
	pos, prob, state = sampler.run_mcmc(p0, burnsteps, progress=True)
	sampler.reset()
	""
	# Production run
	print ("Production run")
	sampler.run_mcmc(pos, nsteps, progress=True)
