import joelib.physics.jethead_expansion as Exp
from joelib.constants.constants import *
from matplotlib.pylab import *
from matplotlib.colors import LogNorm
ion()
import joelib.physics.jethead as Nexp

##### Physical parameters ##############################

# GW170817 parameters according to Lamb et. al for Gaussian structured jet

EE_l   = 10**(52.4)     # Peak core energy
GamC_l = 666.           # Peak core Lorentz factor
epB_l  = 10**(-2.1)
epE_l  = 10**(-1.3)
nn_l   = 10**(-4.1)
pp_l   = 2.16
DD_l   = 41.6e6*pc
thetaCore_l = 0.09
thetaJet_l  = 15.*pi/180
thetaObs_l  = 0.34
nx = 200 # Obsolete parameter, needs to be removed from code
ny = 200 # Obsolete parameter, needs to be removed from code
ncells = 100 # Check if 100 layers is enough
timesteps = 1000 # Check if 500 time steps is enough (should be though)


# Parameters from Zarke, Xie and Macfayden (parameters not clear in the papers), hydro simulations

EE_z   = 10.**(53.)
GamC_z = 100.
epB_z  = 1e-3
epE_z  = 1e-2
nn_z   = 10**(-4.)
pp_z   = 2.15
DD_z   = 40.6e6*pc
thetaCore_z = 0.1
thetaJet_z  = 15.*pi/180


#EE = 1e53; thetaC=0.15; thetaJ = 25*pi/180.; nn = 1e-4; epsE = 0.1; epsB = 0.01; pp=2.16; GamC = 100
EE = 1e52; thetaC=0.15; thetaJ = 15*pi/180.; nn = 1e-3; epsE = 0.1; epsB = 0.01; pp=2.15; GamC = 100
#times = array([8., 10., 15., 25., 50., 60., 75., 85., 100., 120., 150., 175., 200., 250., 300., 350., 400., 450., 500., 550., 600., 700., 800., 900., 1000., 1100., 1200., 1500., 2000., 2500., 3000., 4000., 5000.])*sTd
times = logspace(log10(8), log10(10000), 100)*sTd


#########################################################
# Generate the dynamics

# Jets with constant Sedov length, different central Lorentz factor
jets = [jetHeadGauss(1.e53, gamma, 1e-3, 0.1, 0.01, 2.15, DD_l, timesteps, "peer", ncells, 30.*pi/180, 6.*pi/180) for gamma in range(100,650, 50)]


# Jets with constant central Lorentz factor, varying Sedov length by changing energetics
#jets = [jetHeadGauss(EE, 300., 1e-3, 0.1, 0.01, 2.15, DD_l, timesteps, "peer", ncells, 30.*pi/180, 6.*pi/180) for gamma in logspace(51, 53, 10)]

jets_exp = [Exp.jetHeadGauss(energy, GamC,  nn, epsE, epsB, pp, 2000, 1e15, 1e22, "peer", 150, thetaJ, thetaC, aa=1) for energy in logspace(51, 53, 15)]
jets = [Exp.jetHeadGauss(energy, GamC,  nn, epsE, epsB, pp, 2000, 1e15, 1e22, "peer", 150, thetaJ, thetaC, aa=-1) for energy in logspace(51,53, 15)]

# Jets with constant central Lorentz factor, varying Sedov length by changing medium density
#jets = [jetHeadGauss(1.e52, 300., nn, 0.1, 0.01, 2jets[0].15, DD_l, timesteps, "peer", ncells, 30.*pi/180, 6.*pi/180) for nn in logspace(-2, -4, 10)]


#########################################################



#obsAngles = array([20., 30., 45.])*pi/180.
obsAngles = array([25., 30., 45.])*pi/180
#times = arange(50., 2300., 250)*sTd
times = arange(10., 10000, 100)*sTd



###########################################################

#ttobs_OA, lc_OA, lr_OA = jets[0].lightCurve_interp(0.*pi/180, array([3.0e9, 6.0e9, 4.55e14]), 1e-1, 1000, 200, 1)
#ttobs_15, lc_15, lr_15 = jets[0].lightCurve_interp(15.*pi/180, array([3.0e9, 6.0e9, 4.55e14]), 1e-1, 1000, 200, 1)
#ttobs_30, lc_30, lr_30 = jets[0].lightCurve_interp(30.*pi/180, array([3.0e9, 6.0e9, 4.55e14]), 1e-1, 1000, 200, 1)
#ttobs_195, lc_195, lr_195 = jets[0].lightCurve_interp(0.34, array([3.0e9, 6.0e9, 4.55e14]), 1e-1, 1000, 200, 1)


###########################################################


xx, yy, ff, rr , gams = zeros([len(jets), len(times), len(obsAngles), 2*jets[0].ncells, 2]), zeros([len(jets), len(times), len(obsAngles), 2*jets[0].ncells, 2]
                                       ) , zeros([len(jets), len(times), len(obsAngles), 2*jets[0].ncells, 2]
                                       ) , zeros([len(jets), len(times), len(obsAngles), 2*jets[0].ncells, 2]
                                       ) , zeros([len(jets), len(times), len(obsAngles), 2*jets[0].ncells, 2])
xx_av, yy_av = zeros([len(jets), len(times),len(obsAngles), 2]), zeros([len(jets), len(times), len(obsAngles), 2])
delR = zeros([len(jets), len(times)-1,len(obsAngles), 2])
vv_av = zeros([len(jets), len(times)-1,len(obsAngles), 2])
delT  = diff(times)

ffxmeans, ffymeans = zeros([len(jets), len(times),len(obsAngles), 2]), zeros([len(jets), len(times), len(obsAngles), 2])

for kk in range(len(jets)):
    print("Working on jet")
    for ii in range(len(times)):
        for jj in range(len(obsAngles)):
            ff[kk,ii, jj, :, 0], xx[kk, ii, jj, :, 0], yy[kk, ii, jj, :, 0] , rr[kk, ii, jj, :, 0], gams[kk, ii, jj, :, 0], _ , _ =  Exp.skymapSJ(jets[kk], obsAngles[jj], times[ii], 5e9)

            xx_av[kk, ii, jj, 0], yy_av[kk, ii, jj, 0] = average(xx[kk, ii, jj, :, 0], weights=abs(ff[kk, ii, jj, :, 0])), average(yy[kk, ii, jj, :, 0], weights=abs(ff[kk, ii, jj, :, 0]))

            ff[kk,ii, jj, :, 1], xx[kk, ii, jj, :, 1], yy[kk, ii, jj, :, 1] , rr[kk, ii, jj, :, 1], gams[kk, ii, jj, :, 1], _ , _ =  Exp.skymapSJ(jets_exp[kk], obsAngles[jj], times[ii], 5e9)

            xx_av[kk, ii, jj, 1], yy_av[kk, ii, jj, 1] = average(xx[kk, ii, jj, :, 1], weights=abs(ff[kk, ii, jj, :, 1])), average(yy[kk, ii, jj, :, 1], weights=abs(ff[kk, ii, jj, :, 1]))


for kk in range(len(jets)):
    for jj in range(len(obsAngles)):
        for ii in range(0,len(times)-1):
            delR[kk, ii, jj, 0] = sqrt ((xx_av[kk, ii+1,jj, 0]-xx_av[kk, ii, jj, 0])**2.
                                        + (yy_1av[kk, ii+1, jj, 0]-yy_1av[kk, ii, jj, 0])**2.)
            delR[kk, ii, jj, 1] = sqrt ((xx_av[kk, ii+1,jj, 1]-xx_av[kk, ii, jj, 1])**2.
                                        + (yy_1av[kk, ii+1, jj, 1]-yy_1av[kk, ii, jj, 1])**2.)


for kk in range(len(jets)):
    for ii in range(len(obsAngles)):
        vv_av[kk, :, ii, 0] = delR[kk, :, ii, 0]/delT
        vv_av[kk, :, ii, 1] = delR[kk, :, ii, 1]/delT


"""
fwhm = zeros([len(times), len(obsAngles)])
for ii in range(len(times)):
    for jj in range(len(obsAngles)):
        cs = cumsum(ff_1[ii,jj,:])
        fil = [cs>=0.5]
        fwhm[ii,jj] = yy_1[ii,jj,:][fil][0]

"""
