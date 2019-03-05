from joelib.physics.synchrotron_afterglow import *
from joelib.toolbox.toolBox import uniformLog
import joelib.constants.constants as cts
from matplotlib.pylab import *

ion()


EE   = 1.e45
Gam0 = 100.
nn   = 1.e6
epe  = 0.1
epb  = 1.e-2
pp = 2.5
agtype = "adiabatic"
tt0 = 0.
DD = 40e6*cts.pc

# Initialize the afterglow
mag = afterglow(EE, Gam0, nn, epe, epb, pp, agtype, DD)

tsteps = 5000
flux_points = zeros([tsteps+1, 6])
nuM = zeros([tsteps+1])
nuC = zeros([tsteps+1])

tts = logspace(log10(mag.ttd), log10(500*cts.sTd), num=5000+1, base=10.)
#tts = tts*cts.cts.sTd
nus = logspace(6, 18, num=8000, base=10.)
#nus = 10.**nus
nus_lc = array([1.e6, 1.e8, 1.e12, 1.e14, 1.e16, 1.e18])

tdays = [2.,5.,10.,50.,100., 150., 300.]


fL, aL = subplots()         # Canvas for light curves
fL, aS = subplots()         # Canvas for specta

# Time evolution and catching values for plots later
for ii in range(1,tsteps+1):
    mag.updateAfterGlow(tts[ii])
    flux_points[ii,:] = mag.FluxNuSC(nus_lc)
    nuM[ii] = mag.nuGM
    nuC[ii] = mag.nuCrit

for ii in range(6):
    label = r"$\nu=10^{%1.0d}$ Hz" %round(log10(nus_lc[ii]))
    aL.plot(tts[1:]/cts.sTd, flux_points[1:,ii], label=label)

aL.set_xscale('log')
aL.set_yscale('log')
aL.legend(loc='best', fontsize=10)
aL.tick_params(labelsize=9)

# Calculate spectrum at time instants in tdays
for day in tdays:
    tt = day*cts.sTd
    mag.updateAfterGlow(tt)
    flux = mag.FluxNuSC(nus)
    label = r"$t=%1.0d$ days" %round(day)
    aS.plot(nus, flux, label=label)

aS.set_xscale('log')
aS.set_yscale('log')
aS.legend(loc='best', fontsize=10)
aS.tick_params(labelsize=13)
aS.tick_params(labelsize=9)








"""
nus = arange(log10(1e8), log10(1e18), 0.001)
flux = mag.FluxNuSC(10**nus)
plot(nus, log10(flux))
"""
