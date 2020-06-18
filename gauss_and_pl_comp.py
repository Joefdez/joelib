import joelib.physics.jethead_expansion as Exp
from joelib.constants.constants import *
from matplotlib.pylab import *
from matplotlib.colors import LogNorm
ion()
import joelib.physics.jethead as Nexp
from scipy.interpolate import griddata as gdd


EE = 10**(52.4); GamC = 666; thetaC = 0.09; thetaJ = 30.*pi/180.; thetaObs = 0.34; epsB = 10**(-2.1); epsE = 10**(-1.4); nn0 = 10**(-4.1); pp = 2.16


#jhG    = Exp.jetHeadGauss(EE, GamC,  nn, epsE, epsB, pp, 2000, 1e15, 1e22, "peer", 150, thetaJ, thetaC, aa=1)
#jhPL   = Exp.jetHeadGauss(EE, GamC,  nn, epsE, epsB, pp, 2000, 1e15, 1e22, "peer", 150, thetaJ, thetaC, aa=1, structure='power-law', kk=2)
#jhG_n  = Exp.jetHeadGauss(EE, GamC,  nn, epsE, epsB, pp, 2000, 1e15, 1e22, "peer", 150, thetaJ, thetaC, aa=-1)
#jhPL_n = Exp.jetHeadGauss(EE, GamC,  nn, epsE, epsB, pp, 2000, 1e15, 1e22, "peer", 150, thetaJ, thetaC, aa=-1, structure='power-law', kk=2)

jets = [ Exp.jetHeadGauss(EE, GamC,  nn, epsE, epsB, pp, 2000, 1e15, 1e22, "peer", 150, thetaJ, thetaC, aa=1),
         Exp.jetHeadGauss(EE, GamC,  nn, epsE, epsB, pp, 2000, 1e15, 1e22, "peer", 150, thetaJ, thetaC, aa=1, structure='power-law', kk=2),
         Exp.jetHeadGauss(EE, GamC,  nn, epsE, epsB, pp, 2000, 1e15, 1e22, "peer", 150, thetaJ, thetaC, aa=-1),
         Exp.jetHeadGauss(EE, GamC,  nn, epsE, epsB, pp, 2000, 1e15, 1e22, "peer", 150, thetaJ, thetaC, aa=-1, structure='power-law', kk=2)
]


times = logspace(20, 1000, 25)*sTd

ncells2 = int(2.*jhG.ncells)
ncells = jhG.ncells
nts = len(times)

xxs, yys, ffs, xx_av, yy_av  = zeros([ncells2, nts, 4]), zeros([ncells2, nts, 4]), zeros([ncells2, nts, 4]), zeros([nts, 4]), zeros([nts, 4])
xxs_30, yys_30, ffs_30, xx_av30, yy_av30  = zeros([ncells2, nts, 4]), zeros([ncells2, nts, 4]), zeros([ncells2, nts, 4]), zeros([nts, 4]), zeros([nts, 4])
xxs_45, yys_45, ffs_45, xx_av45, yy_av45  = zeros([ncells2, nts, 4]), zeros([ncells2, nts, 4]), zeros([ncells2, nts, 4]), zeros([nts, 4]), zeros([nts, 4])
#Gams20, Gams30, Gams45  =  zeros([ncells2, nts, 4]),  zeros([ncells2, nts, 4]),  zeros([ncells2, nts, 4])
#RRs20, RRs30, RRs45  =  zeros([ncells2, nts, 4]),  zeros([ncells2, nts, 4]),  zeros([ncells2, nts, 4])


xx_av_PJ, yy_av_PJ, xx_av_CJ, yy_av_CJ  = zeros([nts, 4]), zeros([nts, 4]), zeros([nts, 4]), zeros([nts, 4])
xx_av30_PJ, yy_av30_PJ, xx_av30_CJ, yy_av30_CJ  = zeros([nts, 4]), zeros([nts, 4]), zeros([nts, 4]), zeros([nts, 4])
xx_av45_PJ, yy_av45_PJ, xx_av45_CJ, yy_av45_CJ  = zeros([nts, 4]), zeros([nts, 4]), zeros([nts, 4]), zeros([nts, 4])


for ii in range(nts):
    for jj in range(len(jets)):

        print(ii)

        ffs[:,ii,jj], xxs[:,ii,jj], yys[:,ii,jj], _, _, _, _ = Exp.skymapSJ(jets[ii], 20.*pi/180., times[ii], 5e9)
        ffs_30[:,ii,jj], xxs_30[:,ii,jj], yys_30[:,ii,jj], _, _, _, _ = Exp.skymapSJ(jets[ii], 30.*pi/180., times[ii], 5e9)
        ffs_45[:,ii,jj], xxs_45[:,ii,jj], yys_45[:,ii,jj], _, _, _, _ = Exp.skymapSJ(jets[ii], 30.*pi/180., times[ii], 5e9)
        xx_av[ii,jj] = average(xxs[:,ii,jj], weights=ffs[:,ii,jj])
        xx_av30[ii,jj] = average(xxs_30[:,ii,jj], weights=ffs_30[:,ii,jj])
        xx_av45[ii,jj] = average(xxs_45[:,ii,jj], weights=ffs_45[:,ii,jj])


timesP = array([])
xxsP, yysP, ffsP = zeros([ncells2, 3, 4]), zeros([ncells2, 3, 4]), zeros([ncells2, 3, 4])
xx_avP, yy_avP = zeros([3,4]), zeros([3, 4])
for ii in range(3):
    for jj in range(4):

        ffsP[:,ii,jj], xxsP[:,ii,jj], yysP[:,ii,jj], _, _, _, _ = Exp.skymapSJ(jets[ii], 20.*pi/180., times[ii], 5e9)
