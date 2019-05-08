from joelib.physics.synchrotron_afterglow import adiabatic_afterglow
#from joelib.physics.jethead import *
from joelib.toolbox.toolBox import uniformLog
import joelib.constants.constants as cts
from matplotlib.pylab import *

ion()


EE   = 1.e52
Gam0 = 100.
nn   = 1.
epe  = 0.1
epb  = 1.e-2
pp = 2.5
agtype = "adiabatic"
tt0 = 0.
DD = 40e6*cts.pc

# Initialize the afterglow
mag = adiabatic_afterglow(EE, Gam0, nn, epe, epb, pp, DD)

#tsteps = 1000
rsteps  = 1000
#flux_points = zeros([tsteps+1, 6])
#nuM = zeros([tsteps+1])
#nuC = zeros([tsteps+1])
flux_points = zeros([rsteps+1, 6])
nuM = zeros([rsteps+1])
nuC = zeros([rsteps+1])

#tts = logspace(log10(mag.onaxisTd), log10(200*cts.sTd), num=1000+1, base=10.)
rrs  = logspace(log10(mag.Rd), log10(mag.Rd)+2., num=1000+1, base=10.)
#tts = tts*cts.cts.sTd
nus = logspace(6, 18, num=8000, base=10.)
#nus = 10.**nus
nus_lc = array([1.e6, 1.e8, 1.e12, 1.e14, 1.e16, 1.e18])

#tdays = [2.,5.,10.,50.,100., 150., 300.]
#rr  = mag.onAxisR([2.*cts.sTd,5.*cts.sTd,10.*cts.sTd,50.*cts.sTd,100.*cts.sTd, 150.*cts.sTd, 300.*cts.sTd])
tdays = array([mag.onaxisTd, mag.onaxisTd*2, mag.onaxisTd*10, 2.*cts.sTd,5.*cts.sTd,10.*cts.sTd,50.*cts.sTd,100.*cts.sTd, 150.*cts.sTd, 300.*cts.sTd])
rr = mag.onAxisR(tdays)
fL, aL = subplots()         # Canvas for light curves
fLH, aLH = subplots()         # Canvas for light curves
fL, aS = subplots()         # Canvas for specta


# Time evolution and catching values for plots later
#for ii in range(1,tsteps+1):
for ii in range(1, rsteps+1):
    #mag.updateAG(tts[ii])
    mag.updateAG(rrs[ii])
    if mag.GamMin <= mag.GamCrit:
        flux_points[ii,:] = mag.FluxNuSC(nus_lc)
    else:
        flux_points[ii,:] = mag.FluxNuFC(nus_lc)
    nuM[ii] = mag.nuGM
    nuC[ii] = mag.nuCrit

times = mag.obsTime(theta=0.)

for ii in range(6):
    label = r"$\nu=10^{%1.0d}$ Hz" %round(log10(nus_lc[ii]))
    #aL.plot(tts[1:]/cts.sTd, flux_points[1:,ii], label=label)
    aL.plot(rrs[1:], flux_points[1:,ii], label=label)

aL.set_xscale('log')
aL.set_yscale('log')
aL.legend(loc='best', fontsize=10)
aL.tick_params(labelsize=9)

# Calculate spectrum at different times (in tdays)
for dist in rr:
    #tt = day*cts.sTd
    mag.updateAG(dist)
    label = r"$t=%1.0d$ days" %round(tdays[rr==dist]/cts.sTd)
    print mag.GamMin, mag.GamCrit
    if mag.GamMin <= mag.GamCrit:
        flux = mag.FluxNuSC(nus)
    else:
        flux = mag.FluxNuFC(nus)
    aS.plot(nus, flux, label=label)

aS.set_xscale('log')
aS.set_yscale('log')
aS.legend(loc='best', fontsize=10)
aS.tick_params(labelsize=13)
aS.tick_params(labelsize=9)


"""
jh = jetHeadUD(EE,  Gam0, nn, epe, epb, pp, agtype, DD, 10, 5, 5.)
for ii in range(1,tsteps+1):
    jh.updateJetHead(tts[ii])
    flux_points[ii,:] = jh.totalFluxSCOA(nus_lc)
    #nuM[ii] = mag.nuGM
    #nuC[ii] = mag.nuCrit

for ii in range(6):
    label = r"$\nu=10^{%1.0d}$ Hz" %round(log10(nus_lc[ii]))
    aLH.plot(tts[1:]/cts.sTd, flux_points[1:,ii], label=label)

aLH.set_xscale('log')
aLH.set_yscale('log')
aLH.legend(loc='best', fontsize=10)
aLH.tick_params(labelsize=9)
"""
"""
nus = arange(log10(1e8), log10(1e18), 0.001)
flux = mag.FluxNuSC(10**nus)
plot(nus, log10(flux))
"""
