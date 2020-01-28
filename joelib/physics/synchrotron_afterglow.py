from numpy import *
from scipy.integrate import quad, trapz
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import joelib.constants.constants as cts
from joelib.toolbox.toolBox import rk4


#############################################################################################################


data_xp = genfromtxt('joelib/Xp.dat')

Xint = interp1d(data_xp[:,0], data_xp[:,1])

data_phip = genfromtxt('joelib/PhiP.dat')

PhiPint = interp1d(data_phip[:,0], data_phip[:,1])


#############################################################################################################


def charFreq(gamE, Gam, Bf):
    # Calculate characteristic synchrotron frequency in observer frame

    #BB = (32.*pi*cts.mp*jet.epB*jet.nn)**(1./2.)*jet.Gam*cts.cc
    return Gam*gamE**2.*cts.qe*Bf/(2.*pi*cts.me*cts.cc)
    #return gamE**2.*cts.qe*Bf/(2.*pi*cts.me*cts.cc)


############################################## Sari, Piran & Narayan (1998) formulae ########################


def minGam(Gam, epE, epB, nn, pp, Bfield):
    # Calculate minimum electron lorentz factor and the associated characteristic frequencies

    GamMin   = epE*(pp-2.)/(pp-1.) * cts.mp/cts.me * Gam
    nuGM     = charFreq(GamMin, Gam, Bfield)

    return GamMin, nuGM


def critGam(Gam, epE, epB, nn, pp, Bfield, tt): #, TT, Gam, Bf):
    # Calculate the Critial lorentz factor for efficient synchrotron cooling and associated characteristic frequency

    #GamCrit  = 3.*cts.me/(4.*jet.epB*cts.sigT*cts.mp*jet.nn) * jet.Rd**3./jet.Gam0 * (
    #            RR/jet.Rd)**(9./2.) * 1./(RR**4.+3.*jet.Rd**4.)
    GamCrit =  3.*cts.me/(16.*epB*cts.sigT*cts.mp*cts.cc*Gam**3.*tt*nn)
    nuCrit   = charFreq(GamCrit, Gam, Bfield)



    return GamCrit, nuCrit


def fluxMax(RR, Gam, nn, Bfield, DD):

    # Per unit solid angle!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #fmax = zeros([len(jet.RRs)])
    #fil1 = jet.RRs<=jet.Rd
    #fil2 = jet.RRs>jet.Rd
    #fmax[fil1]  = jet.nn**(3./2.)*jet.RRs[fil1]**3. * cts.sigT * cts.cc**3. *cts.me* (32.*pi*cts.mp*jet.epB
    #                    )**(1./2.)*jet.Gam0**2./(9.*cts.qe*jet.DD**2.)
    #fmax[fil2]  = jet.nn**(3./2.)*jet.Rd**3. * cts.sigT * cts.cc**3. *cts.me* (32.*pi*cts.mp*jet.epB
                            #)**(1./2.)*jet.Gam0**2./(9.*cts.qe*jet.DD**2.)
    fmax  = nn*RR**3. * cts.sigT * cts.cc**2. *cts.me* Bfield* Gam/(9.*4.*pi*cts.qe*DD**2.)

    return fmax


############# Modified expressions to account for transition to Newtonian phase (check with Gavin though!) ##

def Bfield_modified(gam, beta, nn, epB):

    TT = normT(gam, beta)
    ada = adabatic_index(TT)
    eT  = (ada*gam+1.)/(ada-1) * (gam-1.)*nn*cts.mp*cts.cc**2.

    return sqrt(8.*pi*eT*epB)


def minGam_modified(Gam, epE, epB, nn, pp, Bfield, Xp):

    # Modified expression for non-relativistic case

    beta = sqrt(1.-Gam**(-2))

    GamMin    = 1. + cts.mp/cts.me * (pp-2.)/(pp-1.) * epE * (Gam - 1.)
    #Bfield    = sqrt(32. * pi * cts.mp * cts.cc**2. * epB * nn * Gam*(Gam-1))
    nuGM      = 3.*Xp*(Gam*(1.-beta))**(-1)*GamMin**2.*cts.qe*Bfield/(4.*pi*cts.me*cts.cc)

    return GamMin, nuGM


def critGam_modified(Gam, epE, epB, nn, pp, Bfield, tt):

    # Modified expression for non-relativistic case

    beta = sqrt(1.-Gam**(-2))


    #Bfield    = sqrt(32. * pi * cts.mp * cts.cc**2. * epB * nn * Gam*(Gam-1))
    GamCrit   = 6.*pi*cts.me*cts.cc/(cts.sigT*Gam*tt*Bfield**2.)
    nuCrit   = 0.286*3*(Gam*(1.-beta))**(-1.)*GamCrit**2. * cts.qe * Bfield/(4.*pi*cts.me*cts.cc) # Gam factor to convert to observer frame

    return GamCrit, nuCrit


def fluxMax_modified(RR, Gam, nn, Bfield, DD, PhiP):

    #fmax = sqrt(3)*nn*RR**3.*cts.qe**3.*Bfield/(3.*cts.me*cts.cc**2.*4.*pi*DD**2.) *PhiP #
    fmax = sqrt(3)*nn*RR**3.*cts.qe**3.*Bfield/(cts.me*cts.cc**2.*4.*pi*DD**2.) *PhiP #


    return fmax


########### Spectral functions ###############################################################################


def FluxNuSC(jet, nuGM, nuCrit, FnuMax, nu):
    """
    Spectral flux distribution for fast cooling phase.
    Equation 7 in Sari and Piran 1997
    """
    flux = zeros(len(nu))
    #
    #print shape(flux), shape(nuGM), shape(nuCrit), shape(nu) , shape(FnuMax)
    flux[nu<nuGM] = (nu[nu<nuGM]/nuGM)**(1./3.) * FnuMax

    flux[(nu>=nuGM) & (nu<nuCrit)] = (
             (nu[(nu>=nuGM) & (nu<nuCrit)]/nuGM)**(-1.*(jet.pp-1.)/2.) * FnuMax)

    flux[nu>=nuCrit] = (nuCrit/nuGM)**(-1.*(jet.pp-1.)/2.) * (
        nu[nu>=nuCrit]/nuCrit)**(-1.*jet.pp/2.) * FnuMax

    return flux


def FluxNuFC(jet, nuGM, nuCrit, FnuMax, nu):
    """
    Spectral flux distribution for fast cooling phase.
    Equation 8 in Sari and Piran 1997.
    """
    flux = zeros(len(nu))
    flux[nu<nuCrit] = (nu[nu<nuCrit]/nuCrit)**(1./3.) * FnuMax

    flux[(nu>=nuCrit) & (nu<nuGM)] = (
             nu[(nu>=nuCrit) & (nu<nuGM)]/nuCrit)**(-1./2.) * FnuMax

    flux[nu>=nuGM] = (nuGM/nuCrit)**(-1./2.) * (
        nu[nu>=nuGM]/nuGM)**(-jet.pp/2.)*FnuMax

    return flux


def FluxNuSC_arr(jet, nuGM, nuCrit, FnuMax, nu):
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

    flux[fil2] = ((nu[fil2]/nuGM[fil2])**(-1.*(jet.pp-1.)/2.) * FnuMax[fil2])

    flux[fil3] = (nuCrit[fil3]/nuGM[fil3])**(-1.*(jet.pp-1.)/2.) * (
        nu[fil3]/nuCrit[fil3])**(-1.*jet.pp/2.) * FnuMax[fil3]

    return flux


def FluxNuFC_arr(jet, nuGM, nuCrit, FnuMax, nu):
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
        nu[nu>=nuGM]/nuGM[nu>=nuGM])**(-jet.pp/2.)*FnuMax[nu>=nuGM]

    return flux


"""
def obsTime_onAxis(jet, RR):

    return (RR**4.+3.*jet.Rd**4.)/(8.*cts.cc*jet.Rd**3.*jet.Gam0**2.)
"""


#############################################################################################################
############################# Adiabatic, ultrarelativstic dynamics ##########################################
#############################################################################################################


def obsTime_onAxis_adiabatic(RR, Beta):

    "Time calculation following Sari, Piran & Narayn"

    #TTs = jet.RRs/(4.*cts.cc*jet.Gams**2.)

    TTs = RR/(Beta*cts.cc) * (1.-Beta)

    return TTs



def obsTime_offAxis_UR(RR, TT, Beta, theta):
    # Calculate the off-axis time at an angle theta, radius RR and observer time TT
    #tt = obsTime_onAxis(RR)

    #return TT*cos(theta) + RR/cts.cc * (1.-cos(theta))

    return RR/(Beta*cts.cc) * (1.-Beta*cos(theta))



def evolve_ad(jet):
    """
    Evolution following simple energy conservation for an adiabatically expanding relativistic shell. Same scaling as
    Blanford-Mckee blastwave solution. This calculation is only valid in ultrarelativstic phase.
    """
    Gam  = jet.Gam0
    GamSD = 1.021
    Rsd   = Gam**(2./3.) *jet.Rd / GamSD       # Radius at Lorentz factor=1.005 -> after this point use Sedov-Taylor scaling
    Rl    = jet.Rd * Gam**(2./3.)
    RRs   = logspace(log10(jet.Rd/100.), log10(0.9999*Rl), jet.steps+1) #10

    Gams  = zeros(len(RRs))
    Gams[RRs<=jet.Rd] = jet.Gam0
    Gams[RRs>jet.Rd]  = (jet.Rd/RRs[RRs>jet.Rd])**(3./2.) * jet.Gam0
    #Gams[RRs>=Rsd] = 1./sqrt( 1.-(Rsd/RRs[RRs>=Rsd])**(6.)*(1.-1./(Gams[(RRs>jet.Rd) & (RRs<Rsd)][-1]**2.)))
    #Gams[RRs>=jet.Rd] = odeint(jet.dgdr, jet.Gam0, RRs[RRs>=jet.Rd])[:,0]
    #Gams[RRs>=jet.Rd] = odeint(jet.dgdr, jet.Gam0, RRs[RRs>=jet.Rd])[:,0]
    Betas = sqrt(1.-1./Gams**2.)
    Betas[-1] = 0.0


    return RRs, Gams, Betas, Rsd


#############################################################################################################
#############################################################################################################


def timeIntegrand(RR, Rd, Gam0):
    Gam  = (Rd/RR)**(3./2.) * Gam0
    Beta = sqrt(1. - 1./Gam**2.)

    #return (1-Beta)/(cts.cc*Beta)

    return 1./(cts.cc*Gam**2.*Beta*(1.+Beta))


#def timeIntegrand



#############################################################3



#############################################################################################################
######################### Evolution of the blast wave as given in Pe'er 2012 ################################
#############################################################################################################

def obsTime_onAxis_integrated(RRs, Gams, Betas):

    "Very crude numerical integration to obtain the on-axis observer time"
    #timeDiff = (1.-Betas)/(cts.cc*Betas)
    #return cumsum(timeDiff[:-1]*diff(RRs)) + Td

    #fil1 = RRs<=Rd
    #fil2 = (RRs>Rd)

    TTs = zeros(len(Betas))

    #TT  = zeros(len(Betas[fil2]))
    #TTs[fil1] = RRs[fil1]/(
    #                    cts.cc*Beta0*Gam0**2.*(1.+Beta0))

    integrand = 1./(cts.cc*Gams**2.*Betas*(1.+Betas))

    #valf = 1./(cts.cc*Gams[fil1][-1]**2.*Betas[fil1][-1]*(1.+Betas[fil1][-1]))
    #TTs[fil2][0] = trapz(array([RRs[fil1][-1],RRs[fil2][0]]), array([valf, integrand[0]])) +  Td
    TTs[0] = RRs[0]/(cts.cc*Gams[0]**2.*Betas[0]*(1.+Betas[0]))
    for ii in range(1,len(Betas)):
    #for ii in range(1, len(Betas[fil2])):
        #TTs[ii] = quad(timeIntegrand, RRs[ii-1], RRs[ii])[0] + TTs[ii-1]
        TTs[ii] = trapz(integrand[0:ii+1], RRs[0:ii+1]) + TTs[0]

#    TTs[fil2] = TT #+ Td


    return TTs


def obsTime_offAxis_General(RR, TT, theta):

    return TT + RR/cts.cc * (1.-cos(theta))
    #return TT - RR/cts.cc * cos(theta)


def normT(gamma, beta):
    mom = gamma*beta

    return mom/3. * (mom + 1.07*mom**2.)/(1+mom+1.07*mom**2.)

def adabatic_index(theta):
    zz = theta/(0.24+theta)

    return (5-1.21937*zz+0.18203*zz**2.-0.96583*zz**3.+2.32513*zz**4.-2.39332*zz**5.+1.07136*zz**6.)/3.

def dgdm(jet, gam, mm):
    beta = sqrt(1-1./gam**2.)
    TT =   normT(gam, beta)
    ada =  adabatic_index(TT)
    #numerator = -4.*pi*jet.nn*cts.mp*rr**2. * ( ada*(gam**2.-1)-(ada-1)*gam*beta**2  )
    #denominator = jet.EE/(jet.Gam0*cts.cc**2.) + 4./3.*pi*jet.nn*cts.mp*rr**3.*(2.*ada*gam-(ada-1)*(1.+gam**(-2)))
    numerator = -10**mm*log(10)*( ada*(gam**2.-1)-(ada-1)*gam*beta**2  )
    denominator = jet.EE/(jet.Gam0*cts.cc**2.) + 10**mm*(2.*ada*gam-(ada-1)*(1.+gam**(-2)))

    return numerator/denominator

def dgdm_struc(jet, gams, mms):
    betas = sqrt(1-1./gams**2.)
    TTs =   normT(gams, betas)
    adas =  adabatic_index(TTs)
    #numerator = -4.*pi*jet.nn*cts.mp*rr**2. * ( ada*(gam**2.-1)-(ada-1)*gam*beta**2  )
    #denominator = jet.EE/(jet.Gam0*cts.cc**2.) + 4./3.*pi*jet.nn*cts.mp*rr**3.*(2.*ada*gam-(ada-1)*(1.+gam**(-2)))
    numerator = -10**mms*log(10)*( adas*(gams**2.-1)-(adas-1)*gams*betas**2  )
    denominator = jet.cell_EEs/(jet.cell_Gam0s*cts.cc**2.) + 10**mms*(2.*adas*gams-(adas-1)*(1.+gams**(-2)))

    return numerator/denominator

def dgdm_mod(MM0, gam, mm):
    beta = sqrt(1-1./gam**2.)
    TTs =   normT(gam, beta)
    ada =  adabatic_index(TTs)
    #numerator = -4.*pi*jet.nn*cts.mp*rr**2. * ( ada*(gam**2.-1)-(ada-1)*gam*beta**2  )
    #denominator = jet.EE/(jet.Gam0*cts.cc**2.) + 4./3.*pi*jet.nn*cts.mp*rr**3.*(2.*ada*gam-(ada-1)*(1.+gam**(-2)))
    numerator = -10**mm*log(10)*( ada*(gam**2.-1)-(ada-1)*gam*beta**2  )
    denominator = MM0 + 10**mm*(2.*ada*gam-(ada-1)*(1.+gam**(-2)))

    return numerator/denominator




def evolve_relad(jet):
    """
    Evolution following Pe'er 2012. Adbaiatic expansion into a cold, uniform ISM using conservation of energy in relativstic form. This solution
    transitions smoothly from the ultra-relativistic to the Newtonian regime.
    """
    adas = zeros(jet.steps+1)
    Gam  = jet.Gam0
    GamSD = 1.021
    Rsd   = Gam**(2./3.) *jet.Rd / GamSD       # Radius at Lorentz factor=1.005 -> after this point use Sedov-Taylor scaling
    Rl    = jet.Rd * Gam**(2./3.)
    RRs   = logspace(log10(jet.Rd/100.), log10(Rl)+1.5, jet.steps+1) #10
    #MMs = 1./3. * RRs**3. * jet.nn * cts.mp * jet.angExt
    MMs   = 4./3. *pi*cts.mp*jet.nn*RRs**3.

    Gams  = zeros(len(RRs))
    Gams[0] = Gam

    for ii in range(1,jet.steps+1):
        Gams[ii] = rk4(dgdm, jet, log10(MMs[ii-1]), Gams[ii-1], (log10(MMs[ii])-log10(MMs[ii-1])))


    Betas = sqrt(1.-1./Gams**2.)
    #Betas[-1] = 0.0

    return RRs, Gams, Betas, Rsd






#############################################################################################################
#############################################################################################################
#############################################################################################################

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



class adiabatic_afterglow:

    """
    Simple afterglow model assuming adiabatic evolution
    """

    def __init__(self, EE, Gam0, nn, epE, epB, pp, DD, steps, evolution, shell_type, Rb):
        self.EE   = EE
        self.Gam0 = Gam0
        self.Beta0 = (1.-1./self.Gam0**2.)**(1./2.)
        self.nn   = nn
        self.epE  = epE
        self.epB  = epB
        self.pp   = pp
        self.Xp   = Xint(pp)
        self.PhiP = PhiPint(pp)
        self.DD   = DD
        self.steps = steps
        self.evolution = evolution
        self.shell_type = shell_type
        self.Rb = Rb
        self.__decRad()
        self.__onAxisdecTime()
        #self.updateAG(self.Rd)
        self.__evolve()
        self.__fluxMax()
        self.FnuMI = interp1d(self.RRs, self.FnuMax)
        self.__peakParamsRS()

#        self.__obsTime_onAxis()
# ========== Private methods =======================================================================================================


    def __decRad(self):
        # Deceleration radius of the jet

        Beta0 = sqrt(1.-1./self.Gam0**(2.))
        #if self.evolution == "adiabatic":
        #    self.Td = self.Rd/(2.*cts.cc*self.Gam0**2.)
        #elif self.evolution == "peer":
        #    self.Td = self.Rd*(1.-Beta0)/(cts.cc*Beta0)
        #self.Td = self.Rd/(2.*self.Gam0**2 * cts.cc)
        if self.evolution == "adiabatic":
            self.Rd = (3./(4.*pi) * 1./(cts.cc**2.*cts.mp) *
                            self.EE/(self.nn*self.Gam0**2.))**(1./3.)
            self.Td = self.Rd/(cts.cc*self.Beta0) * (1.-self.Beta0)
        elif self.evolution == "peer":
            self.Rd = (3./(4.*pi) * 1./(cts.cc**2.*cts.mp) *
                            self.EE/(self.nn*self.Gam0**2.))**(1./3.)
            self.Td = self.Rd*(1.-Beta0)/(cts.cc*Beta0)



    def __onAxisdecTime(self):
        # Deceleration time for an on-axis observer
        self.onaxisTd = self.Rd/(2.*self.Gam0**2 * cts.cc)
        #self.onaxisTd = (3.*self.EE/(256*pi*self.nn*self.Gam0**8. *cts.mp*cts.cc**5 ))**(1./3)


#    def __obsTime_onAxis(self):
#        self.TTs = self.obsTime_onAxis()

    def __fluxMax(self):
        # Calculate maximum spectral flux -> Units energy/(time freq area)
        # (to be multplied by the solid angle covered by the segment) and divided by 1/(4 pi D**2)
        if self.evolution == 'adiabatic':
            self.FnuMax = fluxMax(self.RRs, self.Gams, self.nn, self.Bfield, self.DD)
        elif self.evolution == 'peer':
            self.FnuMax = fluxMax_modified(self.RRs, self.Gams, self.nn, self.Bfield, self.DD, self.PhiP)


    def __evolve(self):

        if self.evolution == "adiabatic":
            self.RRs, self.Gams, self.Betas, self.Rsd = evolve_ad(self)
            self.Bfield = (32.*pi*cts.mp*self.epB*self.nn)**(1./2.)*self.Gams*cts.cc

        elif self.evolution == "peer":
            self.RRs, self.Gams, self.Betas, self.Rsd = evolve_relad(self)
            self.Bfield  = Bfield_modified(self.Gams, self.Betas, self.nn, self.epB)
            #self.Bfield = (32.*pi*cts.mp*self.epB*self.nn*self.Gams*(self.Gams-1))**(1./2.)*cts.cc

        self.GamInt = interp1d(self.RRs, self.Gams)

        if self.evolution == "adiabatic":
            self.TTs = obsTime_onAxis_adiabatic(self.RRs, self.Betas)
        elif self.evolution == "peer":
            #self.TTs = obsTime_onAxis_integrated(self)
            self.TTs = obsTime_onAxis_integrated(self.RRs, self.Gams, self.Betas)

        #self.onAxisTint = interp1d(self.RRs, self.TTs)
        if self.evolution == 'adiabatic':
            self.gM, self.nuM = minGam(self.Gams, self.epE, self.epB, self.nn, self.pp, self.Bfield)
            self.gC, self.nuC = critGam(self.Gams, self.epE, self.epB, self.nn, self.pp, self.Bfield, self.TTs)
        elif self.evolution == 'peer':
            self.gM, self.nuM = minGam_modified(self.Gams, self.epE, self.epB, self.nn, self.pp, self.Bfield, self.Xp)
            self.gC, self.nuC = critGam_modified(self.Gams, self.epE, self.epB, self.nn, self.pp, self.Bfield, self.TTs)

        self.gamMI, self.gamCI = interp1d(self.RRs, self.gM), interp1d(self.RRs, self.gC)
        self.nuMI, self.nuCI = interp1d(self.RRs, self.nuM), interp1d(self.RRs, self.nuC)

    def __peakParamsRS(self):

        Gam0 = self.Gam0

        # These need to be scaled be the correspinding factor of Rb when calculating light curve

        if self.shell_type=='thin':
            print("Settig up thin shell")
            #self.RSpeak_nuM = 9.6e14 * epE**2. * epB**(1./2.) * nn**(1./2) * Gam0**2.
            #self.RSpeak_nuC = 4.0e16 * epB**(-3./2.) * EE**(-2./3.) * nn**(-5./6.) * Gam0**(4./3.)
            #self.RSpeak_Fnu = 5.2 * DD**(-2.) * epB**(1./2.) * EE * nn**(1./2.) * Gam0
            self.RSpeak_nuM  = self.nuMI(self.Rd)/(Gam0**2) #* self.Rb**(1./2.)
            self.RSpeak_nuC  = self.nuCI(self.Rd) #* self.Rb**(-3./2.)*
            self.RSpeak_Fnu =  Gam0*self.FnuMI(self.Rd)# * self.Rb**(1./2.)*



#=====================================================================================================================================================#
#=====================================================================================================================================================#
#=====================================================================================================================================================#
