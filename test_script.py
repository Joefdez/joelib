from joelib.physics.synchrotron_afterglow import adiabatic_afterglow
from joelib.physics.synchrotron_afterglow import  afterglow
from joelib.constants.constants import *
from numpy import *
from matplotlib.pylab import *

# Generate instances of the two models
aga = afterglow(1e52, 100, 1., 0.1, 0.01, 2.5, "adiabatic", 1e6*pc)
ada = adiabatic_afterglow(1e52, 100, 1., 0.1, 0.01, 2.5, 1e6*pc)

# Obtain the radii, corresponding to an expansion of ~300 days according to the adiabatic model
adaRrs  = logspace(log10(ada.Rd), log10(ada.Rd)+3., num=1000, base=10.)

# Obtain the corresponding apparent times in the distant observer frame  according to the adiabatic model.
# Use to evaluate the evolution in the SPN model
adaTts = (adaRrs**4.+3.*ada.Rd**4.)/(8.*cc*ada.Rd**3.*ada.Gam0**2.)

# Frequencies for light curves
nus_lc = array([1.e6, 1.e8, 1.e10, 1.e12, 1.e14, 1.e16, 1.e18])


# Empty arrays to be filled during numerical evolution
ada_nuM = zeros([1000])
ada_nuC = zeros([1000])
ada_Gam = zeros([1000])
ada_fmax = zeros([1000])
ada_flux_points = zeros([1000, 7])


aga_nuM = zeros([1000])
aga_nuC = zeros([1000])
aga_Rad = zeros([1000])
aga_Gam = zeros([1000])
aga_fmax = zeros([1000])
aga_flux_points = zeros([1000, 7])


# Evolution block

for ii in range(1000):
    aga.updateAfterGlow(adaTts[ii])
    aga_nuM[ii] = aga.nuGM
    aga_nuC[ii] = aga.nuCrit
    aga_Rad[ii] = aga.Rad
    aga_Gam[ii] = aga.fgam
    aga_fmax[ii] = aga.FnuMax

    if aga.GamMin <= aga.GamCrit:
        aga_flux_points[ii,:] = aga.FluxNuSC(nus_lc)
    else:
        aga_flux_points[ii,:] = aga.FluxNuFC(nus_lc)


    ada.updateAG(adaRrs[ii])
    ada_nuM[ii] = ada.nuGM
    ada_nuC[ii] = ada.nuCrit
    ada_Gam[ii] = ada.Gam
    ada_fmax[ii] = ada.FnuMax

    if ada.GamMin <= ada.GamCrit:
        ada_flux_points[ii,:] = ada.FluxNuSC(nus_lc)
    else:
        ada_flux_points[ii,:] = ada.FluxNuFC(nus_lc)
