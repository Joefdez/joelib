from numpy import *
from numpy.linalg import solve, det
from numpy.random import uniform
import joelib.physics.jethead_expansion as Exp
from joelib.constants.constants import *
from scipy.interpolate import interp1d
import emcee

def priors(parameters, ranges):
    # Returns log of prior distribution of parameters with limits on ranges
    # For priors which are uniform in linear space or log space.
    # parameters and ranges must be input with appropriate scaling (e.g. in log scale or linear scale)

    vals = []

    for parameter, range in zip(parameters, ranges):

        if range[0]<parameter<range[1]:
            vals.append(range[0]-range[1])
        else:
            vals.append(-1.*inf)

    return array(vals)


def afterglow_likelihood_gaussianJet(model_parameters, data_lc):

    # model_parameters --> the input to generate the model
    # data_parameters  --> the parameters used to calculate the comparison data (like freqneucies, times, etc)
    # data --> the observational data itself
    # data_err --> errors on the data

    EE, GamC, thetaC, thetaObs, epsB, epsE, nn, pp  = model_parameters   # Unpack model parameters
    EE, nn, epsE, epsB = 10.**(EE), 10.**(nn), 10.**(epsE), 10.**(epsB)
    #print(EE, GamC, thetaC, thetaObs, epsB, epsE, nn, pp)
    nparams = len(model_parameters)


    delxmu = array([])
    data_err = array([])

    #lld = 0 # initialize log-likelihood

    jet = Exp.jetHeadGauss(EE, GamC, nn, epsE, epsB, pp, 500, 1e14, 1e22, 'peer', 50, 30.*pi/180, thetaC, aa=1)
    #print jet.TTs.min()
    freqs = array(data_lc.keys()).astype('float')
    tts, lcModel, _, _ = Exp.light_curve_peer_SJ(jet, jet.pp, thetaObs, freqs, 41.3e6*pc, "range", [1.,1000.,100], 1)           # Calculte the model prediction at the observation time

    #maskD = [dataType=="lightCurve"]
    #lcfreqs, lcdata, lcdataErr = frequencies[maskD], data[maskD] #, data_err[maskD]

    #maskC = [dataType=='centShift']
    #centShiftFreqs, centShiftdata, centShiftErr = frequencies[maskC] #, data[maskC], data_err[maskC]


    for ii, freq in zip(range(len(freqs)), data_lc.keys()):

        times, lc, lc_err = data_lc[freq][0,:], data_lc[freq][1,:], data_lc[freq][2,:]                              # Get the observational times and data

        #_, lcModel, _, _ = Exp.light_curve_peer_SJ(jet, jet.pp, thetaObs, freq, 41.3e6*pc, "discrete", times*sTd, 1)           # Calculte the model prediction at the observation time
        lcM = interp1d(tts, lcModel[ii,:])(times*sTd)*1e29
        delxmu = append(delxmu, lcM-lc)
        data_err = append(data_err, lc_err)


    ffs75, xxs75, _, _, _, _, _ = Exp.skymapSJ(jet, thetaObs, 75*sTd, 4.5e9)
    ffs230, xxs230, _, _, _, _, _ = Exp.skymapSJ(jet, thetaObs, 230*sTd, 4.5e9)

    xcModel75, xcModel230 = average(xxs75, weights=ffs75), average(xxs75, weights=ffs230)
    xcModel = (xcModel230-xcModel75)*rad2mas/(DD)
    delxmu = append(delxmu, (xcModel-2.7))
    data_err = append(data_err, ones(1)*0.3)

    covMatrix = diag(data_err*data_err)



    loglikelihood = -0.5*dot(delxmu, solve(covMatrix,delxmu)) -0.5*nparams*log(2.*pi) - 0.5*log(det(covMatrix))

    return loglikelihood



def log_probability(parameters, ranges, lc_dict):
    lp = priors(parameters, ranges)

    #if not isfinite(lp):
    #    return -1.*inf

    return lp + afterglow_likelihood_gaussianJet(parameters, lc_dict)




def initialize_walkers(nwalkers, nparameters, ranges):

    initializations = zeros([nwalkers, nparameters])

    for ii in range(nparameters):
        initializations[:,ii] = uniform(ranges[ii][0], ranges[ii][1], nwalkers)


    return initializations



data = genfromtxt("gw170817_data.txt", dtype='str')
data = data[data[:,-1].astype('float')>0]
ranges = [[51., 53.], [100.,1000.], [0.01, 0.1], [0.3, 0.4], [-4.,-0.5], [-4.,-0.5], [-5., 0.], [2.01, 2.25]]
DD = 41.3e6*pc

freqs = unique(data[:,3])

nwalkers = 2
nparameters = len(ranges)

lc_dict = {}

for freq in freqs:
     fil = data[:,3]==freq
     fil2 = data[:,-1][fil].astype('float')>0.
     if len(data[:,0][fil][fil2])!=0:
         lc_dict[freq] = array([data[:,1][fil][fil2].astype("float"), data[:,4][fil][fil2].astype("float"), data[:,-1][fil][fil2].astype("float")])



p0 = initialize_walkers(nwalkers, nparameters, ranges)
sampler = emcee.EnsembleSampler(nwalkers, nparameters, log_probability, args=[ranges, lc_dict])

# Burn-in steps
state = sampler.run_mcmc(p0, 10, progress=True)
sampler.reset()

# Run MCMC
#sampler.run_mcmc(state, 10000)
