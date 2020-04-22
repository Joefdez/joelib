from numpy import *
from joelib.constants.constants import cc, mp, me, qe, sigT
from scipy.interpolate import interp1d

#############################################################################################################


data_xp = genfromtxt('joelib/Xp.dat')

Xint = interp1d(data_xp[:,0], data_xp[:,1])

data_phip = genfromtxt('joelib/PhiP.dat')

PhiPint = interp1d(data_phip[:,0], data_phip[:,1])


##############################################################################################################



################################################################# Physical quantities ########################################################################################



def normT(gamma, beta):
    mom = gamma*beta

    return mom/3. * (mom + 1.07*mom**2.)/(1+mom+1.07*mom**2.)

def adabatic_index(theta):
    zz = theta/(0.24+theta)

    return (5-1.21937*zz+0.18203*zz**2.-0.96583*zz**3.+2.32513*zz**4.-2.39332*zz**5.+1.07136*zz**6.)/3.


def charFreq(gamE, Gam, Bf):
    # Calculate characteristic synchrotron frequency in observer frame

    #BB = (32.*pi*mp*jet.epB*jet.nn)**(1./2.)*jet.Gam*cc
    return Gam*gamE**2.*qe*Bf/(2.*pi*me*cc)
    #return gamE**2.*qe*Bf/(2.*pi*me*cc)

#################################################################################################################################################################

def Bfield_modified(gam, beta, nn, epB):

    TT = normT(gam, beta)
    ada = adabatic_index(TT)
    eT  = (ada*gam+1.)/(ada-1) * (gam-1.)*nn*mp*cc**2.

    return sqrt(8.*pi*eT*epB)


def minGam_modified(Gam, epE, epB, nn, pp, Bfield, Xp):



    beta = sqrt(1.-Gam**(-2))
    GamMin    = mp/me * (pp-2.)/(pp-1.) * epE * (Gam - 1.)
    #Bfield    = sqrt(32. * pi * mp * cc**2. * epB * nn * Gam*(Gam-1))
    nuGM      = 3.*Xp*(Gam*(1.-beta))**(-1)*GamMin**2.*qe*Bfield/(4.*pi*me*cc)
    #nuGM      = 3.*(Gam*(1.-beta))*Xp*GamMin**2.*qe*Bfield/(4.*pi*me*cc)
    return GamMin, nuGM


def critGam_modified(Gam, epE, epB, nn, pp, Bfield, tt):


    beta = sqrt(1.-Gam**(-2))


    #Bfield    = sqrt(32. * pi * mp * cc**2. * epB * nn * Gam*(Gam-1))
    GamCrit   = 6.*pi*me*cc/(sigT*Gam*tt*Bfield**2.)
    nuCrit   = 0.286*3*(Gam*(1.-beta))**(-1.)*GamCrit**2. * qe * Bfield/(4.*pi*me*cc) # Gam factor to convert to observer frame
    #nuCrit   = 0.286*3.*(Gam*(1.-beta))*GamCrit**2. * qe * Bfield/(4.*pi*me*cc) # Gam factor to convert to observer frame

    return GamCrit, nuCrit


def fluxMax_modified(RR, Gam, Ne, Bfield, PhiP):        # Flux max * D**2.

    #fmax = sqrt(3)*nn*RR**3.*qe**3.*Bfield/(3.*me*cc**2.*4.*pi*DD**2.) *PhiP #
    fmax = sqrt(3)*Ne*qe**3.*Bfield/(me*cc**2.*4.*pi) *PhiP #

    return fmax

#########################################################################################################################################################################


def params_tt_RS(rs, tt, Rb):

    if type(tt) == 'float': tt = array([tt])
    fil1, fil2 = where(tt<=rs.Td)[0], where(tt>rs.Td)[0]

    nuM = zeros(len(tt))
    nuC = zeros(len(tt))
    fluxMax = zeros(len(tt))

    nuM[fil1] = rs.RSpeak_nuM*(tt[fil1]/rs.Td)**(6.)
    nuC[fil1] = rs.RSpeak_nuC*(tt[fil1]/rs.Td)**(-2.)
    fluxMax[fil1] = rs.RSpeak_Fnu*(tt[fil1]/rs.Td)**(3./2.)

    nuM[fil2]  = rs.RSpeak_nuM*(tt[fil2]/rs.Td)**(-54./35.)
    nuC[fil2]  = rs.RSpeak_nuC*(tt[fil2]/rs.Td)**(4./35.)
    fluxMax[fil2] = rs.RSpeak_Fnu*(tt[fil2]/rs.Td)**(-34./35.)

    """
    if tt<rs.Td:
        nuM  = rs.RSpeak_nuM*(tt/rs.Td)**(6.)
        nuC  = rs.RSpeak_nuC*(tt/rs.Td)**(-2.)
        fluxMax = rs.RSpeak_Fnu*(tt/rs.Td)**(3./2.)   # Returns fluxes in Jy
    else:
        nuM  = rs.RSpeak_nuM*(tt/rs.Td)**(-54./35.)
        nuC  = rs.RSpeak_nuC*(tt/rs.Td)**(4./35.)
        fluxMax = rs.RSpeak_Fnu*(tt/rs.Td)**(-34./35.) # Returns fluxes in Jy
    """

    return Rb**(1./2.) * nuM, Rb**(-3./2.)*nuC, Rb**(1./2.)*fluxMax


def params_tt_RS_SJ(jet, tt, ii, Rb):

        if type(tt) == 'float': tt = array([tt])
        fil1, fil2 = where(tt<=jet.cell_Tds[ii])[0], where(tt>jet.cell_Tds[ii])[0]
        #print ii, len(tt)
        nuM = zeros(len(tt))
        nuC = zeros(len(tt))
        fluxMax = zeros(len(tt))
        #print len(nuM), len(nuC), len()

        nuM[fil1]     = jet.RSpeak_nuM_struc[ii]*(tt[fil1]/jet.cell_Tds[ii])**(6.)
        nuC[fil1]     = jet.RSpeak_nuC_struc[ii]*(tt[fil1]/jet.cell_Tds[ii])**(-2.)
        fluxMax[fil1] = jet.RSpeak_Fnu_struc[ii]*(tt[fil1]/jet.cell_Tds[ii])**(3./2.)   # Returns fluxes in Jy

        nuM[fil2]     = jet.RSpeak_nuM_struc[ii]*(tt[fil2]/jet.cell_Tds[ii])**(-54./35.)
        nuC[fil2]     = jet.RSpeak_nuC_struc[ii]*(tt[fil2]/jet.cell_Tds[ii])**(4./35.)
        fluxMax[fil2] = jet.RSpeak_Fnu_struc[ii]*(tt[fil2]/jet.cell_Tds[ii])**(-34./35.) # Returns fluxes in Jy

        return Rb**(1./2.)*nuM, Rb**(-3./2.)*nuC, Rb**(1./2.)*fluxMax



#########################################################################################################################################################################




def FluxNuSC(pp, nuGM, nuCrit, FnuMax, nu):
    """
    Spectral flux distribution for fast cooling phase.
    Equation 7 in Sari and Piran 1997
    """
    flux = zeros(len(nu))
    #
    #print shape(flux), shape(nuGM), shape(nuCrit), shape(nu) , shape(FnuMax)
    flux[nu<nuGM] = (nu[nu<nuGM]/nuGM)**(1./3.) * FnuMax

    flux[(nu>=nuGM) & (nu<nuCrit)] = (
             (nu[(nu>=nuGM) & (nu<nuCrit)]/nuGM)**(-1.*(pp-1.)/2.) * FnuMax)

    flux[nu>=nuCrit] = (nuCrit/nuGM)**(-1.*(pp-1.)/2.) * (
        nu[nu>=nuCrit]/nuCrit)**(-1.*pp/2.) * FnuMax

    return flux


def FluxNuFC(pp, nuGM, nuCrit, FnuMax, nu):
    """
    Spectral flux distribution for fast cooling phase.
    Equation 8 in Sari and Piran 1997.
    """
    flux = zeros(len(nu))
    flux[nu<nuCrit] = (nu[nu<nuCrit]/nuCrit)**(1./3.) * FnuMax

    flux[(nu>=nuCrit) & (nu<nuGM)] = (
             nu[(nu>=nuCrit) & (nu<nuGM)]/nuCrit)**(-1./2.) * FnuMax

    flux[nu>=nuGM] = (nuGM/nuCrit)**(-1./2.) * (
        nu[nu>=nuGM]/nuGM)**(-pp/2.)*FnuMax

    return flux


def FluxNuSC_arr(pp, nuGM, nuCrit, FnuMax, nu):
    """
    Spectral flux distribution for fast cooling phase.
    Equation 7 in Sari and Piran 1997
    """
    flux = zeros(len(nu))
    #print shape(flux), shape(nuGM), shape(nuCrit), shape(nu) , shape(FnuMax)
    fil1 = nu<nuGM
    fil2 = (nu>=nuGM) & (nu<nuCrit)
    fil3 = nu>=nuCrit

    #print "SC", shape(nuGM), shape(nuCrit), shape(FnuMax)
    flux[fil1] = (nu[fil1]/nuGM[fil1])**(1./3.) * FnuMax[fil1]

    flux[fil2] = ((nu[fil2]/nuGM[fil2])**(-1.*(pp-1.)/2.) * FnuMax[fil2])

    flux[fil3] = (nuCrit[fil3]/nuGM[fil3])**(-1.*(pp-1.)/2.) * (
        nu[fil3]/nuCrit[fil3])**(-1.*pp/2.) * FnuMax[fil3]

    return flux


def FluxNuFC_arr(pp, nuGM, nuCrit, FnuMax, nu):
    """
    Spectral flux distribution for fast cooling phase.
    Equation 8 in Sari and Piran 1997.
    """
    flux = zeros(len(nu))

    #fil1 =

    #print "FC", shape(nuGM), shape(nuCrit), shape(FnuMax)
    flux[nu<nuCrit] = (nu[nu<nuCrit]/nuCrit[nu<nuCrit])**(1./3.) * FnuMax[nu<nuCrit]

    flux[(nu>=nuCrit) & (nu<nuGM)] = (
        nu[(nu>=nuCrit) & (nu<nuGM)]/nuCrit[(nu>=nuCrit) & (nu<nuGM)])**(-1./2.) * FnuMax[(nu>=nuCrit) & (nu<nuGM)]

    flux[nu>=nuGM] = (nuGM[nu>=nuGM]/nuCrit[nu>=nuGM])**(-1./2.) * (
        nu[nu>=nuGM]/nuGM[nu>=nuGM])**(-pp/2.)*FnuMax[nu>=nuGM]

    return flux
