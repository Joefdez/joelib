from numpy import *
from scipy.integrate import quad, trapz
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import joelib.constants.constants as cts
from joelib.toolbox.toolBox import rk4




class adiabatic_afterglow:

    """
    Simple afterglow model assuming adiabatic evolution
    """

    def __init__(self, EE, Gam0, nn, epE, epB, pp, DD, steps, evolution):
        self.EE   = EE
        self.Gam0 = Gam0
        self.Beta0 = (1.-1./self.Gam0**2.)**(1./2.)
        self.nn   = nn
        self.epE  = epE
        self.epB  = epB
        self.pp   = pp
        self.DD   = DD
        self.steps = steps
        self.evolution = evolution
        self.__decRad()
        self.__onAxisdecTime()
        #self.updateAG(self.Rd)
        self.__evolve()
        self.__fluxMax()
        self.FnuMI = interp1d(self.RRs, self.FnuMax)
#        self.__obsTime_onAxis()
# ========== Private methods =======================================================================================================


    def __decRad(self):
        # Deceleration radius of the jet
        self.Rd = (3./(4.*pi) * 1./(cts.cc**2.*cts.mp) *
                        self.EE/(self.nn*self.Gam0**2.))**(1./3.)
        Beta0 = sqrt(1.-1./self.Gam0**(2.))
        self.Td = self.Rd*(1.-Beta0)/(cts.cc*Beta0)
        #self.Td = self.Rd/(2.*self.Gam0**2 * cts.cc)


    def __onAxisdecTime(self):
        # Deceleration time for an on-axis observer
        self.onaxisTd = self.Rd/(2.*self.Gam0**2 * cts.cc)

#    def __obsTime_onAxis(self):
#        self.TTs = self.obsTime_onAxis()

    def __fluxMax(self):
        # Calculate maximum spectral flux -> Units energy/(time freq area)
        # (to be multplied by the solid angle covered by the segment) and divided by 1/(4 pi D**2)

        self.FnuMax = self.fluxMax()

    def __evolve(self):

        if self.evolution == "adiabatic":
            self.RRs, self.Gams, self.Betas, self.Rsd = self.evolve_ad()
        elif self.evolution == "peer":
            self.RRs, self.Gams, self.Betas, self.Rsd = self.evolve_relad()

        self.GamInt = interp1d(self.RRs, self.Gams)
        self.TTs = self.obsTime_onAxis()
        self.Bfield = (32.*pi*cts.mp*self.epB*self.nn)**(1./2.)*self.Gams*cts.cc
        self.gM, self.nuM = self.minGam()
        self.gC, self.nuC = self.critGam()
        self.gamMI, self.gamCI = interp1d(self.RRs, self.gM), interp1d(self.RRs, self.gC)
        self.nuMI, self.nuCI = interp1d(self.RRs, self.nuM), interp1d(self.RRs, self.nuC)



    """

    def __onAxisTime(self):
        # Observer time for on-axis observer (minus the on-axis deceleration time)
        # So the code basically computes T-Td
        self.onaxisTT = (self.RR**4.+3.*self.Rd**4.)/(8.*cts.cc*self.Rd**3.*self.Gam0**2.)  #+ self.onaxisTD


    def __minGam(self, RR):
        # Calculate minimum electron Lorentz factor at current R and
        # corresponding synchrotron characteristic frequency

        self.GamMin, self.nuGM = self.minGam(self.RR)


    def __critGam(self, RR):
        # Calculate critical Lorentz factor for efficient cooling at R and
        # corresponding synchrotron characteristic frequency

        self.GamCrit, self.nuCrit = self.critGam(self.RR)


    def __Gam(self):
        # Calculate the bulk Lorentz factor as a function of the radius of the shell
        self.Gam = (self.Rd/self.RR)**(3./2.) * self.Gam0
    """


# ========== Public methods ==========================================================================================================



    def charFreq(self, gamE, Gam, Bf):
        # Calculate characteristic synchrotron frequency in observer frame

        #BB = (32.*pi*cts.mp*self.epB*self.nn)**(1./2.)*self.Gam*cts.cc
        return Gam*gamE**2.*cts.qe*Bf/(2.*pi*cts.me*cts.cc)


    #def onAxisR(self, tt):
        # Calculate radius of the shell as a function of the on-axis observer time
    #    return (8.* cts.cc*self.Gam0**2. *self.Rd**3.*tt - 3.*self.Rd**4.)**(1./4.)


    def minGam(self):
        # Calculate minimum electron lorentz factor and the associated characteristic frequencies

        GamMin   = self.epE*(self.pp-2.)/(self.pp-1.) * cts.mp/cts.me * self.Gams
        nuGM     = self.charFreq(GamMin, self.Gams, self.Bfield)

        return GamMin, nuGM

    def critGam(self): #, TT, Gam, Bf):
        # Calculate the Critial lorentz factor for efficient synchrotron cooling and associated characteristic frequency

        #GamCrit  = 3.*cts.me/(4.*self.epB*cts.sigT*cts.mp*self.nn) * self.Rd**3./self.Gam0 * (
        #            RR/self.Rd)**(9./2.) * 1./(RR**4.+3.*self.Rd**4.)
        GamCrit =  3.*pi*cts.me/(16.*self.epB*cts.sigT*cts.mp*cts.cc*self.Gams**3.*self.TTs*self.nn)
        nuCrit   = self.charFreq(GamCrit, self.Gams, self.Bfield)

        return GamCrit, nuCrit

    def fluxMax(self):

        # Per unit solid angle!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #fmax = zeros([len(self.RRs)])
        #fil1 = self.RRs<=self.Rd
        #fil2 = self.RRs>self.Rd
        #fmax[fil1]  = self.nn**(3./2.)*self.RRs[fil1]**3. * cts.sigT * cts.cc**3. *cts.me* (32.*pi*cts.mp*self.epB
        #                    )**(1./2.)*self.Gam0**2./(9.*cts.qe*self.DD**2.)
        #fmax[fil2]  = self.nn**(3./2.)*self.Rd**3. * cts.sigT * cts.cc**3. *cts.me* (32.*pi*cts.mp*self.epB
                            #)**(1./2.)*self.Gam0**2./(9.*cts.qe*self.DD**2.)
        fmax  = self.nn**(3./2.)*self.RRs**3. * cts.sigT * cts.cc**3. *cts.me* (32.*pi*cts.mp*self.epB
                             )**(1./2.)*self.Gams**2./(9.*cts.qe*self.DD**2.)
        return fmax


    def FluxNuSC(self, nuGM, nuCrit, FnuMax, nu):
        """
        Spectral flux distribution for fast cooling phase.
        Equation 7 in Sari and Piran 1997
        """
        flux = zeros(len(nu))
        #
        #print shape(flux), shape(nuGM), shape(nuCrit), shape(nu) , shape(FnuMax)
        flux[nu<nuGM] = (nu[nu<nuGM]/nuGM)**(1./3.) * FnuMax

        flux[(nu>=nuGM) & (nu<nuCrit)] = (
                 (nu[(nu>=nuGM) & (nu<nuCrit)]/nuGM)**(-1.*(self.pp-1.)/2.) * FnuMax)

        flux[nu>=nuCrit] = (nuCrit/nuGM)**(-1.*(self.pp-1.)/2.) * (
            nu[nu>=nuCrit]/nuCrit)**(-1.*self.pp/2.) * FnuMax

        return flux


    def FluxNuFC(self, nuGM, nuCrit, FnuMax, nu):
        """
        Spectral flux distribution for fast cooling phase.
        Equation 8 in Sari and Piran 1997.
        """
        flux = zeros(len(nu))

        flux[nu<nuCrit] = (nu[nu<nuCrit]/nuCrit)**(1./3.) * FnuMax

        flux[(nu>=nuCrit) & (nu<nuGM)] = (
                 nu[(nu>=nuCrit) & (nu<nuGM)]/nuCrit)**(-1./2.) * FnuMax

        flux[nu>=nuGM] = (nuGM/nuCrit)**(-1./2.) * (
            nu[nu>=nuGM]/nuGM)**(-self.pp/2.)*FnuMax

        return flux


    def FluxNuSC_arr(self, nuGM, nuCrit, FnuMax, nu):
        """
        Spectral flux distribution for fast cooling phase.
        Equation 7 in Sari and Piran 1997
        """
        flux = zeros(len(nu))
        #print shape(flux), shape(nuGM), shape(nuCrit), shape(nu) , shape(FnuMax)
        fil1 = nu<nuGM
        fil2 = (nu>=nuGM) & (nu<nuCrit)
        fil3 = nu>=nuCrit

        flux[fil1] = (nu[fil1]/nuGM[fil1])**(1./3.) * FnuMax[fil1]

        flux[fil2] = ((nu[fil2]/nuGM[fil2])**(-1.*(self.pp-1.)/2.) * FnuMax[fil2])

        flux[fil3] = (nuCrit[fil3]/nuGM[fil3])**(-1.*(self.pp-1.)/2.) * (
            nu[fil3]/nuCrit[fil3])**(-1.*self.pp/2.) * FnuMax[fil3]

        return flux


    def FluxNuFC_arr(self, nuGM, nuCrit, FnuMax, nu):
        """
        Spectral flux distribution for fast cooling phase.
        Equation 8 in Sari and Piran 1997.
        """
        flux = zeros(len(nu))

        #fil1 =


        flux[nu<nuCrit] = (nu[nu<nuCrit]/nuCrit)**(1./3.) * FnuMax

        flux[(nu>=nuCrit) & (nu<nuGM)] = (
            nu[(nu>=nuCrit) & (nu<nuGM)]/nuCrit)**(-1./2.) * FnuMax

        flux[nu>=nuGM] = (nuGM/nuCrit)**(-1./2.) * (
            nu[nu>=nuGM]/nuGM)**(-self.pp/2.)*FnuMax

        return flux


    """
    def obsTime_onAxis(self, RR):

        return (RR**4.+3.*self.Rd**4.)/(8.*cts.cc*self.Rd**3.*self.Gam0**2.)
    """

    def timeIntegrand(self, RR):
        Gam  = (self.Rd/RR)**(3./2.) * self.Gam0
        Beta = sqrt(1. - 1./Gam**2.)

        #return (1-Beta)/(cts.cc*Beta)

        return 1./(cts.cc*Gam**2.*Beta*(1.+Beta))


    #def timeIntegrand

    def obsTime_onAxis(self):

        "Very crude numerical integration to obtain the on-axis observer time"
        #timeDiff = (1.-self.Betas)/(cts.cc*self.Betas)
        #return cumsum(timeDiff[:-1]*diff(self.RRs)) + self.Td

        #fil1 = self.RRs<=self.Rd
        #fil2 = (self.RRs>self.Rd)

        TTs = zeros(len(self.Betas))

        #TT  = zeros(len(self.Betas[fil2]))
        #TTs[fil1] = self.RRs[fil1]/(
        #                    cts.cc*self.Beta0*self.Gam0**2.*(1.+self.Beta0))

        integrand = 1./(cts.cc*self.Gams**2.*self.Betas*(1.+self.Betas))

        #valf = 1./(cts.cc*self.Gams[fil1][-1]**2.*self.Betas[fil1][-1]*(1.+self.Betas[fil1][-1]))
        #TTs[fil2][0] = trapz(array([self.RRs[fil1][-1],self.RRs[fil2][0]]), array([valf, integrand[0]])) +  self.Td
        TTs[0] = self.RRs[0]/(cts.cc*self.Gams[0]**2.*self.Betas[0]*(1.+self.Betas[0]))
        for ii in range(1,len(self.Betas)):
        #for ii in range(1, len(self.Betas[fil2])):
            #TTs[ii] = quad(self.timeIntegrand, self.RRs[ii-1], self.RRs[ii])[0] + TTs[ii-1]
            TTs[ii] = trapz(integrand[0:ii+1], self.RRs[0:ii+1]) + TTs[0]

    #    TTs[fil2] = TT #+ self.Td


        return TTs



        return TTs

    def obsTime_offAxis(self, RR, TT, theta):
        # Calculate the off-axis time at an angle theta, radius RR and observer time TT
        #tt = obsTime_onAxis(RR)

        #return TT*cos(theta) + RR/cts.cc * (1.-cos(theta))

        return TT + RR/cts.cc * (1.-cos(theta))



    # Evolution of the blast wave as given in Pe'er 2012

    def normT(self, gamma, beta):
        mom = gamma*beta

        return mom/3. * (mom + 1.07*mom**2.)/(1+mom+1.07*mom**2.)

    def adabatic_index(self, theta):
        zz = theta/(0.24+theta)

        return (5-1.21937*zz+0.18203*zz**2.-0.96583*zz**3.+2.32513*zz**4.-2.39332*zz**5.+1.07136*zz**6.)/3.

    def dgdm(self, gam, mm):
        beta = sqrt(1-1./gam**2.)
        TT = self.normT(gam, beta)
        ada = self.adabatic_index(TT)
        #numerator = -4.*pi*self.nn*cts.mp*rr**2. * ( ada*(gam**2.-1)-(ada-1)*gam*beta**2  )
        #denominator = self.EE/(self.Gam0*cts.cc**2.) + 4./3.*pi*self.nn*cts.mp*rr**3.*(2.*ada*gam-(ada-1)*(1.+gam**(-2)))
        numerator = -( ada*(gam**2.-1)-(ada-1)*gam*beta**2  )
        denominator = self.EE/(self.Gam0*cts.cc**2.) + mm*(2.*ada*gam-(ada-1)*(1.+gam**(-2)))

        return numerator/denominator


    #############################################################3


    def evolve_ad(self):
        """
        Evolution following simple energy conservation for an adiabatically expanding relativistic shell. Same scaling as
        Blanford-Mckee blastwave solution. This calculation is only valid in ultrarelativstic phase.
        """
        Gam  = self.Gam0
        GamSD = 1.021
        Rsd   = Gam**(2./3.) *self.Rd / GamSD       # Radius at Lorentz factor=1.005 -> after this point use Sedov-Taylor scaling
        Rl    = self.Rd * Gam**(2./3.)
        RRs   = logspace(log10(self.Rd/10.), log10(Rl), self.steps+1) #10

        Gams  = zeros(len(RRs))
        Gams[RRs<=self.Rd] = self.Gam0
        Gams[RRs>self.Rd]  = (self.Rd/RRs[RRs>self.Rd])**(3./2.) * self.Gam0
        #Gams[RRs>=Rsd] = 1./sqrt( 1.-(Rsd/RRs[RRs>=Rsd])**(6.)*(1.-1./(Gams[(RRs>self.Rd) & (RRs<Rsd)][-1]**2.)))
        #Gams[RRs>=self.Rd] = odeint(self.dgdr, self.Gam0, RRs[RRs>=self.Rd])[:,0]
        #Gams[RRs>=self.Rd] = odeint(self.dgdr, self.Gam0, RRs[RRs>=self.Rd])[:,0]
        Betas = sqrt(1.-1./Gams**2.)
        Betas[-1] = 0.0


        return RRs, Gams, Betas, Rsd


    def evolve_relad(self):
        """
        Evolution following Pe'er 2012. Adbaiatic expansion into a cold, uniform ISM using conservation of energy in relativstic form. This solution
        transitions smoothly from the ultra-relativistic to the Newtonian regime.

        """
        Gam  = self.Gam0
        GamSD = 1.021
        Rsd   = Gam**(2./3.) *self.Rd / GamSD       # Radius at Lorentz factor=1.005 -> after this point use Sedov-Taylor scaling
        Rl    = self.Rd * Gam**(2./3.)
        RRs   = logspace(log10(self.Rd/100.), log10(Rl)+3., self.steps) #10
        MMs    = 4./3. *pi*cts.mp*self.nn*RRs**3.

        Gams  = zeros(len(RRs))
        Gams[0] = Gam

        for ii in range(1,self.steps):
            Gams[ii] = rk4(self.dgdm, MMs[ii], Gams[ii-1], (MMs[ii]-MMs[ii-1]))


        Betas = sqrt(1.-1./Gams**2.)
        #Betas[-1] = 0.0

        return RRs, Gams, Betas, Rsd




#=====================================================================================================================================================#
#=====================================================================================================================================================#
#=====================================================================================================================================================#


class afterglow:

#E, Gam0, nn, epE, epB, pp, tt0, agtype):

    """
    Afterglow class for studying the time evolution of GRB afterglow.
    As described in Sari and Piran 1997
    Atributes:
        EE -> total energy of the shock
        nn -> ambient electron density
        epE -> fraction of shock energy density in electrons
        epB -> fraction of schock energy density in magnetic field
        ttd -> initial observer time, set to the decelaration time
        tt  -> time variable
        pp  -> spectral index of electron Lorentz factor distribution
        GamMin -> minimum Lorentz factor of electron distribution

    """

    def __init__(self, EE,  Gam0, nn, epE, epB, pp, agtype, DD):
        """
        Initialize the afterglow at observer time tt0 = decelarion time
        """
        self.EE = EE                                                                  # Total energy of shock
        #self.tt  = tt0                                                               # Current time
        self.Gam0 = Gam0                                                              # Lorentz factor of shocked fluid
        self.Gam  = Gam0
        self.MM = self.EE/(self.Gam0*cts.cc**2.)                                      # Ejecta mass
        self.nn = nn                                                                  # Ambient electron density
        self.epE = epE                                                                # Fraction of energy of shock transfered to electrons
        self.epB = epB                                                                # Energy density of magnetic field, in units of shock energy
        self.pp  = pp                                                                 # Spectral index of electrom Lorentz factor distributuion
        self.DD = DD
        self.GamMin = self.epE*(self.pp-2.)/(self.pp-1.) * cts.mp/cts.me *self.Gam    # Minimum Lorentz factor of electron distribution
        self.LL = (17.*self.MM/(16.*pi*cts.mp*self.nn))**(1./3.)                      # Sedov length

        # Obtain deceleration radius time and initialize time variable
        ttd = (3./(32.*pi) * 1./(cts.cc**5.*cts.mp*self.nn) *self.Gam0**(-8.) * self.EE)**(1./3.)

        self.ttd = ttd
        self.tt  = ttd           # Initialize time variable
        if (agtype == "adiabatic") or (agtype == "radiative"):
            #pass
            self.agtype = agtype
        else:
            print("Wrong afterglow type. Only use adiabatic or radiative.")
            self.agtype = None

        self.__criticalGam()
        self.__selfTransTime()
        if self.agtype == "adiabatic":
            self.__adiabaticEvolution()
        elif self.agtype == "radiative":
            self.__radiativeEvolution()

        self.__shellRadius()
        self.__transFreq()
        #self.__criticalTimes()


###############################################################################################
# Methods for initializing afterglow time parameters and time evolution
###############################################################################################

    def __charFreq(self, gamE):
        """
        Characteristic synchrotron frequency in the observer frame.
        Equation 4 in Sari and Piran 1997
        """
        return self.Gam*gamE**2. * cts.qe*self.BB/(2.*pi*cts.me*cts.cc)


    def __criticalGam(self):

        self.GamCrit = 3.*cts.me/(16.*self.epB *cts.sigT*cts.mp
                                )*1./(cts.cc*self.tt*self.Gam**3. *self.nn)

    def __minGam(self):
        """
        Minimusm Lorentz factor of the electrons.
        Equation 1 in Sari and Piran 1997
        """
        return self.epE*(self.pp-2.)/(self.pp-1.) * cts.mp/cts.me *self.gam


    def __shellRadius(self):
        """
        Compute radius of expanding shell ater time tt.
        Equation 9 in Sari and Piran 1997
        """
        if self.agtype == "adiabatic":
            self.Rad = (17.*self.EE*self.tt/(4.*pi*cts.mp*self.nn*cts.cc))**(1./4.)
        elif self.agtype == "radiative":
            self.Rad = (4.*cts.cc*self.tt/self.LL)**(1./7.) * LL

    def __fluidGam(self):
        """
        Compute shocked fluid Lorentz factor after a time tt .
        Equation 10 in Sari and Piran 1997
        """
        if self.agtype == "adiabatic":
            self.fgam = (17.*self.EE/(1024.*pi*cts.mp*self.nn*cts.cc**5.*self.tt**3.))**(1./8.)
        elif self.agtype == "radiative":
            self.fgam = (4.*cts.cc*self.tt/self.LL)**(-3./7.)

    def __selfTransTime(self):
        """
        Calculate transition time between fast and slow cooling in days.
        Equation 13 in Sari and Piran 1997
        """

        tt = self.tt/cts.sTd          # time in days
        gam = self.Gam/100            # Lorentz factor in unit of 100
        EE = self.EE/1.e52            # In units of 10**52 ergs
        DD = self.DD/1.e26            # In units of 10**28 cm
        #nn = self.nn/1.e6             # In units of cm**3.
        nn = self.nn
        if self.agtype == "adiabatic":
            self.ttrans = 210.*(self.epB * self.epE)**2. * EE * nn
        elif self.agtype == "radiative":
            self.ttrans = 4.6*(self.epB * self.epE)**(7./5.) * gam**(-4./5.)*nn**(3./5.)

    def __adiabaticEvolution(self):
        """
        Time evolution of critical frequencies and maximum spectral flux in the adiabatic case.
        Equation 11 in Sari and Piran 1997
        """

        tt = self.tt/cts.sTd          # time in days
        EE = self.EE/1.e52            # In units of 10**52 ergs
        DD = self.DD/1.e28            # In units of 10**28 cm
        #nn = self.nn/1.e6             # In units of cm**3.
        nn = self.nn
        self.nuCrit = 2.7e12*self.epB**(-3./2.)*EE**(-1./2.)*nn**(-1.)*tt**(-1./2.)
        self.nuGM   = 5.7e14*self.epB**(1./2.)*self.epE**2.*EE**(1./2.)*tt**(-3./2.)
        self.FnuMax  = 1.1e5*self.epB**(1./2.)*EE*nn**(1./2.)*DD**(-2.) # In micro Janskys (extra factor of 1e6 when compared to paper)


    def __radiativeEvolution(self):
        """
        Time evolution of critical frequencies and maximum spectral flux in the adiabatic case.
        Equation 12 in Sari and Piran 1997
        """

        tt = self.tt/cts.sTd      # time in days
        gam = self.Gam/100.       # Lorentz factor in units of 100
        EE = self.EE/1.e52           # In units of 10**52 ergs
        DD = self.DD/1.e28        # In units of 10**28 cm
        #nn = self.nn/1.e6         # In units of cm**3
        nn = self.nn
        self.nuCrit = 1.3e13*self.epB**(-3./2.)*EE**(-4./7.)*nn**(-13./14)*tt**(-2./7.)
        self.nuGM   = 1.2e14*self.epB**(1./2.)*self.epE**2.*EE**(4./7.)*nn**(-1./14)*tt**(-12./7.)
        self.FnuMax  = 4.5e3*self.epB**(1./2.)*EE**(8./7.)*gam**(-8./7.)*nn**(5./14.)*DD**(-2.)*tt**(-3./7.) # In Joules (extra factor of 1e6 when compared to paper)



    def __transFreq(self):
        """
        Critical frequency at which nu0 = nuCrit(tt0) = nuGM(tt0)
        """
        gam = self.Gam/100.
        EE = self.EE/1.e52   # In units of 10**52 ergs
        #nn = self.nn/1.e6    # In units of cm**3
        nn = self.nn
        if self.agtype == "adiabatic":
            self.nuT = 1.8e11*self.epB**(-5./2.) * self.epE**(-1.)*EE**(-1.)*nn**(-3./2.)
        elif self.agtype == "radiative":
            self.nuT = 8.5e12*self.epB**(-19./10.) * self.epE**(-2./5.)*EE**(-4./5.)*gam**(4./5.)*nn**(-11./10.)



###############################################################################################
# Methods for extracting fluxes, light-curves, etc.
###############################################################################################

    def updateAfterGlow(self, tt):
        """
        Update the dynamical properties of the afterglow
        """
        self.tt = tt
        self.__shellRadius()
        self.__fluidGam()
        self.__criticalGam()
        self.__shellRadius()
        if self.agtype == "radiative":
            self.__radiativeEvolution()
        elif self.agtype == "adiabatic":
            self.__adiabaticEvolution()
        else:
            print("Wrong afterglow type. Only use adiabatic or radiative.")

    def synchrotronPower(self, gamE):
        """
        Synchrotron power in the observer frame.
        Equation 3 in Sari and Piran 1997
        """
        return 4./3. * cts.sigT * cts.cc * self.Gam**2. * gamE**2. * self.BB**2./(8.*cts.pi)



    def FluxNuSC(self, nu):
        """
        Spectral flux distribution for fast cooling phase.
        Equation 7 in Sari and Piran 1997
        """
        flux = zeros(len(nu))

        flux[nu<self.nuGM] = (nu[nu<self.nuGM]/self.nuGM)**(1./3.) * self.FnuMax

        flux[(nu>=self.nuGM) & (nu<self.nuCrit)] = (
                 (nu[(nu>=self.nuGM) & (nu<self.nuCrit)]/self.nuGM)**(-1.*(self.pp-1.)/2.) * self.FnuMax)

        flux[nu>=self.nuCrit] = (self.nuCrit/self.nuGM)**(-1.*(self.pp-1.)/2.) * (
            nu[nu>=self.nuCrit]/self.nuCrit)**(-1.*self.pp/2.) * self.FnuMax

        return flux


    def FluxNuFC(self, nu):
        """
        Spectral flux distribution for fast cooling phase.
        Equation 8 in Sari and Piran 1997.
        """
        flux = zeros(len(nu))

        flux[nu<self.nuCrit] = (nu[nu<self.nuCrit]/self.nuCrit)**(1./3.) * self.FnuMax

        flux[(nu>=self.nuCrit) & (nu<self.nuGM)] = (
                 nu[(nu>=self.nuCrit) & (nu<self.nuGM)]/self.nuCrit)**(-1./2.) * self.FnuMax

        flux[nu>=self.nuGM] = (self.nuGM/self.nuCrit)**(-1./2.) * (
            nu[nu>=self.nuGM]/self.nuGM)**(-self.pp/2.)*self.FnuMax

        return flux


    def criticalTimes(self, nu):
        """
        Times at which the frequencies associated with the critical and minumum Lorentz factors cross the observed
        frequency nu
        Two outputs, tc and tm, in days
        """
        nu = nu/(1.e15)
        gam = self.Gam/100.
        EE = self.EE/1.e52   # In units of 10**52 ergs
        #nn = self.nn/1.e6    # In units of cm**3
        nn = self.nn
        if self.agtype == "adiabatic":
            ttc = 7.3e-6*self.epB**(-3.)*EE**(-1.)*nn**(-2.)*nu**(-2.)
            ttm = 0.69*self.epB**(1./3.)*self.epE**(4./3.)*EE**(1./3.)*nu**(-2./3)
        elif self.agtype == "radiative":
            ttc = 2.7e-7*self.epB**(-21./4.)*EE**(-2.)*gam**(2.)*nn**(-13./4.)*nu**(-7./2)
            ttm = 0.29*self.epB**(7./24.)*self.epE**(7./6.)*EE*(1./3.)*gam**(-1./3.)*nn**(-1./24)*gam**(4./5.)

        return ttc, ttm
