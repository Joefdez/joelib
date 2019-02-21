from numpy import pi
import joelib.constants.constants as cts



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
        tt0 -> initial observer time
        pp  -> spectral index of electron Lorentz factor distribution
        GamMin -> minimum Lorentz factor of electron distribution
    """

    def __init__(self, EE,  Gam0, nn, epE, epB, pp, tt0, agtype):
        """
        Initialize the afterglow at observer time tt0
        """
        self.EE = EE                                                          # Total energy of shock
        self.tt  = tt0                                                         # Current time
        self.Gam0 = Gam0                                                      # Lorentz factor of shocked fluid
        self.Gam  = Gam0
        self.MM = self.EE/(self.Gam0*cts.cc**2.)                                  # Ejecta mass
        self.nn = nn                                                          # Ambient electron density
        self.epE = epE                                                        # Fraction of energy of shock transfered to electrons
        self.epB = epB                                                        # Energy density of magnetic field, in units of shock energy
        self.pp  = pp                                                         # Spectral index of electrom Lorentz factor distributuion
        self.GamMin = self.epE*(self.pp-2.)/(self.pp-1.) * cts.mp/cts.me *self.Gam    # Minimum Lorentz factor of electron distribution
        self.LL = (17.*self.MM/(16.*pi*cts.mp*self.nn))**(1./3.)

        if (agtype == "adiabatic") or (agtype == "radiative"):
            pass
            self.agtype = agtype
        else:
            print("Wrong afterglow type. Only use adiabatic or radiative.")
            self.agtype = None
            #self.GamCrit = 3.*me/(16.*self.epB *sigT*mp
            #               )*1./(cc*tt0*Gam**3. *nn)    # Critical Lorentz factor at t=tt0

        self.GamCrit = 3.*cts.me/(16.*self.epB *cts.sigT*cts.mp
                       )*1./(cts.cc*self.tt*self.Gam**3. *nn)

        if self.agtype == "adiabatic":
            self.ttrans = 210.*(self.epB * self.epE)**2. * (self.EE/1E52) * nn
        elif self.agtype == "radiative":
            self.ttrans = 4.6*(self.epB * self.epE)**(-7./5.) * (self.Gam/1.E2)**(-4./5.)*(nn*1e6)**(3./5.)

    def criticalGam(self):
        """
        Compute the critical Lorentz factor at time tt at which cooling by Synchrotron becomes significant
        Equation 6 in Sari and Piran 1997
        """
        return 3.*cts.me/(16.*self.epB *cts.sigT*cts.mp
                       )*1./(cts.cc*self.tt*Gam**3. *nn)

    def minGam(self):
        """
        Minimum Lorentz factor of the electrons.
        Equation 1 in Sari and Piran 1997
        """
        return self.epE*(self.pp-2.)/(self.pp-1.) * cts.mp/cts.me *self.gam

    def synchrotronPower(self, gamE):
        """
        Synchrotron power in the observer frame.
        Equation 3 in Sari and Piran 1997
        """
        return 4./3. * cts.sigT * cts.cc * self.Gam**2. * gamE**2. * self.BB**2./(8.*cts.pi)

    def charFreq(self, gamE):
        """
        Characteristic synchrotron frequency in the observer frameself.
        Equation 4 in Sari and Piran 1997
        """
        return self.Gam*gamE**2. * cts.qe*self.BB/(2.*pi*cts.me*cts.cc)

    def updateAfterGlow(self, tt):

        self.GamCrit = criticalGam(self)

    def FluxNuFC(self, nu):
        """
        Spectral flux distribution for fast cooling phaseself.
        Equation 7 in Sari and Piran 1997
        """
        if self.nuCrit > nu :
            flux = (nu/self.nuCrit)**(1./3.) * self.FnuMax
        elif (nu > self.nuCrit) and (self.num > nu):
            flux = (nu/self.nuCrit)**(-1./2.) * self.FnuMax
        elif nu> self.num:
            flux = (self.num/self.nuCrit)**(-1./2.) * (
                 nu/self.num)**(-self.pp/2.)*self.FnuMax
        return flux

    def FluxNuSC(self, nu):
        """
        Spectral flux distribution for fast cooling phase.
        Equation 8 in Sari and Piran 1997.
        """
        if self.nuCrit > nu :
            flux = (nu/self.nuCrit)**(1./3.) * self.FnuMax
        elif (nu > self.nuCrit) and (self.num > nu):
            flux = (nu/self.nuCrit)**(-1./2.) * self.FnuMax
        elif nu> self.num:
            flux = (self.num/self.nuCrit)**(-1./2.) * (
                nu/self.num)**(-self.pp/2.)*self.FnuMax
        return flux

    def shellRadius(self, tt):
        """
        Compute radius of expanding shell ater time tt.
        Equation 9 in Sari and Piran 1997
        """
        if self.agtype == "adiabatic":
            self.Rad = (17.*EE*tt/(4.*pi*cts.mp*nn*cts.cc))**(1./4.)
        elif self.agtype == "radiative":
            self.Rad = (4.*cts.cc*tt/LL)**(1./7.) * LL

    def fluidGam(self, tt):
        """
        Compute shocked fluid Lorentz factor after a time tt .
        Equation 10 in Sari and Piran 1997
        """
        if self.agtype == "adiabatic":
            self.fgam = (17.*EE/(1024.*pi*cts.mp*nn*cts.cc**5.*tt**3.))**(1./8.)
        elif self.agtype == "radiative":
            self.fgam = (4.*cts.cc*tt/LL)**(-3./7.)

    def selfTransTime(self, nn1):
        """
        Calculate transition time between fast and slow cooling in days.
        Equation 13 in Sari and Piran 1997
        """

        gam = self.Gam/100.  # Lorentz factor in units of 100
        EE = self.EE/1.e52
        nn = self.nn*1.e6    # In units of cm**3.

        if self.agtype == "adiabatic":
            self.ttrans = 210.*(self.epB * self.epE)**2. * EE * nn
        elif self.agtype == "radiative":
            self.ttrans = 4.6*(self.epB * self.epE)**(-7./5.) * gam**(-4./5.)*nn**(3./5.)

    def adiabaticEvolution(self, tt):
        """
        Time evolution of critical frequencies and maximum spectral flux in the adiabatic case.
        Equation 11 in Sari and Piran 1997
        """
        self.tt = tt         # update time
        tt = tt/sTd          # time in days
        EE = self.EE/1.e52   # In units of 10**52 ergs
        DD = self.DD/1.e26   # In units of 10**28 cm
        nn = self.nn*1.e6    # In units of cm**3.

        self.nuCrit = 2.7e12*self.epB**(-3./2.)*EE**(-1./2.)*nn**(-1.)*tt**(-1./2.)
        self.nuGM   = 5.7e14*self.epB**(1./2.)*self.epE**2.*EE**(1./2.)*tt**(-3./2.)
        self.Fvmax  = 1.1e5*self.epB**(1./2.)*EE*nn**(1./2.)*DD**(-2.)


    def radiativeEvolution(self, tt):
        """
        Time evolution of critical frequencies and maximum spectral flux in the adiabatic case.
        Equation 12 in Sari and Piran 1997
        """
        self.tt = tt         # update time
        tt = tt/sTd          # time in days
        gam = self.Gam/100.  # Lorentz factor in units of 100
        EE = self.EE/1.e52   # In units of 10**52 ergs
        DD = self.DD/1.e26   # In units of 10**28 cm
        nn = self.nn*1.e6    # In units of cm**3

        self.nuCrit = 1.3e13*self.epB**(-3./2.)*EE**(-4./7.)*nn**(-13./14)*tt**(-2./7.)
        self.nuGM   = 1.2e14*self.epB**(1./2.)*self.epE**2.*EE**(4./7.)*nn**(-1./14)*tt**(-12./7.)
        self.Fvmax  = 4.5e3*self.epB**(1./2.)*EE**(8./7.)*gam**(-8./7.)*nn**(5./14.)*DD**(-2.)*tt**(-3./7.)



    def criticalTimes(self, nu):
        """
        Times at which the frequencies associated with the critical and minumum Lorentz factors cross the observed
        frequency nu
        Two outputs, tc and tm, in days
        """
        nu = nu/(1.e15)
        gam = self.Gam/100.
        EE = self.EE/1.e52   # In units of 10**52 ergs
        nn = self.nn*1.e6    # In units of cm**3

        if self.agtype == "adiabatic":
            tc = 7.3e-6*self.epB**(-3.)*EE**(-1.)*nn**(-2.)*nu**(-2.)
            tm = 0.69*self.epB**(1./3.)*self.epE**(4./3.)*EE**(1./3.)*nu**(-2./3)
        elif self.agtype == "radiative":
            tc = 2.7e-7*self.epB**(-21./4.)*EE**(-2.)*gam**(2.)*nn**(-13./4.)*nu**(-7./2)
            tm = 0.29*self.epB**(7./24.)*self.epE**(7./6.)*EE*(1./3.)*gam**(-1./3.)*nn**(-1./24)*gam**(4./5.)
