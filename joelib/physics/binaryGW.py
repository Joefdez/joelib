from binarySystem import *

class binaryGW:

    def __init__(self):
        pass

    def chirpMass(m1, m2):

        return (m1*m2)**(3./5.)/(m1+m2)**(1./5.)

    def circStrain(m1, m2, ff, DD):
        # Sourced from  ref:
        cM = chirpMass(m1, m2)
        return 1./DD * 2.*(4.*pi)**(1./3.)*GG**(5./3.)/cc**4. * ff**(2./3.) * cM**(5./3.)

    def peakFreqEcc_approx(mm1, mm2, aa, ee):
        """
        Compute the peak gravitational wave frequency
        """
        orbF = freq(mm1, mm2, aa)
        return 2.*orbF*(1.-ee)**(-3./2.)

#    def peakFreqEcc(mm1, mm2, aa, ee):

#        orbF = freq(mm1, mm2, aa)
#        return 2.*orbF*(1.-ee)**(1.1954)/(1.-ee**2.)**(3./2.)


    def eccF(ee):
        """
        Enhancement factor to the power emitted in GWs by a binary of eccentricity ee
        with respect to the circular counterpart.
        """
        return (1. + (73./24.) * ee**2. + (37./96.) * ee**4.)/((1.-ee**2.)**(7./2.))

    def harmG(self, nn, ee):
        """
        Returns the relative power emitted in the harmonic nn of the binary orbital frequency
        given an orbital eccentricity ee.
        """
        ne = nn*ee
        cc1 = jv(nn-2, ne) - 2.*ee*jv(nn-1,ne) + 2./nn * jv(nn, ne) + 2.*ee*jv(nn+1, ne) - jv(nn+2, ne)
        cc2 = (1.-ee**2.) * (jv(nn-2,ne) - 2.*jv(nn, ne) + jv(nn+2, ne))**2.
        cc3 = 4./(3.*nn**2.) * jv(nn,ne)**2.

        return (nn**4.)/32. * (cc1**2. + cc2 + cc3)

    def dEdf(self, cMass, ff, zz, ee, nn):

        gg = harmG(nn, ee)
        enhancement = eccF(ee)
        preFac =  (GG*cMass)**(5./3.)/(3.*pi**(1./3.)*(1.+ zz)**(1./3.)*ff**(1./3.))

        return preFac*(2./nn)**(2./3.) * gg/enhancement

    def eccEVinitcond(aa0, ee0):
        """
        Initial condition coefficient in sma evolution as a function of eccentricity with
        initial conditions aa0 and ee0.
        """
        cc0 = (1-ee0**2.)/(ee0**(12./19.)) *(1.+121./304. * ee0**2.)**(-870./2299.) *aa0

        return cc0

    def evolution_sma(aa, ee, cc0):
        """
        Semi-major axis
        """

        aa = cc0 *ee**(12./19.)/(1.-ee**2.) * (1. + 121./304. *ee**2.)**(870./2299.)

        return aa


bgw = binaryGW()
