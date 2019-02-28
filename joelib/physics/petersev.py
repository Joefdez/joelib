class PetersEv:
    """
    Class containing functions for analysing the evolution of compact binaries due to
    gravitational wave radiation, as prescribed by Peter's equations
    """

    def __init__(self):
        pass

    def cc0Factor(self, aa0, ee0):
        # Integration constant for Peter's equations
        c1  = aa0*(1.-ee0**2.)*ee0**(-12./19.)
        c2 = 1. + (121./304.)*(ee0**2.)

        return c1*(c2**(-870./2299.))

    def betaFactor(self, m1, m2):
        # Beta factor of binary of componentmasses m1, m2

        return (64./5.)*GG**3. * m1*m2*(m1+m2)/(cc**5)


    def mergerTimeInt(self, ee):
    # Integral of merger time formula of a compact binary object due to GW radiation according to Peters equations:
    # "Gravitational Radiation and the Motion of Two Point Masses", Phys. Rev. 136 4b, 1964
    # ee: eccentricity

        num = ee**(29./19.) * (1. + (121./304.)*ee**2.)**(1181./2299.)
        den = (1. - ee**2)**(3./2.)

        return num/den

    def mergerTimeecc(self, aa0s, ee0s, betaf):
        # Merger time formula of an eccentric compact binary object due to GW radiation according to Peters equations:
        # "Gravitational Radiation and the Motion of Two Point Masses", Phys. Rev. 136 4b, 1964
        # Assumes equal mass binaries (i.e single value of beta factor)
        # ee0s: array of initial eccentricities
        # aa0s: array of initial semi-major axes

        numS = shape(aa0s)
        cc0s = cc0Factor(aa0s, ee0s)

        prefactor = (12./19.) * cc0s**4. / betaf

        ints = quadArray(mergerTimeInt, zeros(numS), ee0s)

        return prefactor*ints

    def mergerTimecirc(self, aa0s, beta):
        # Merger time of a circular binary ue to GW radiation according to Peters equations:
        # "Gravitational Radiation and the Motion of Two Point Masses", Phys. Rev. 136 4b, 1964
        # Assumes equal mass binaries (i.e single value of beta factor)
        # aa0s: array of initial semi-major axes

        return aa0s**4./beta

    def minSMA(self, ees, tt, beta):
        # semi-major axis at which given an eccentricity ee the binary with beta factor beta will merge in a time tt
        # Inputs are asrrays, at least ee.

        numS = shape(ees)
        ints = quadArray(mergerTimeInt, zeros(numS), ees)

        prefactor = ees**(12./19.)/(1.-ees**2.) * (1. + (121./304.)*(ees**2.))**(870./2299.)

        return prefactor*(19./12. * tt*beta/ints)**(1./4.)

    def timeIntGWev(self, cc0, betaF, ee1, ees):
        # Compute the time inteval between two GW evolutionary states for a binary, given
        # the initial semi-major axis, eccentricity and masses of components

        nn = shape(ees)[0]
        ee1 = ee1*ones(nn)

        return 12./19. * cc0**4./betaF * quadArray(mergerTimeInt, ees, ee1)


    def orbFreqEcc(self, ee, ff0, ee0):
        # Compute orbital frequency as a function of eccentricity, given the initial orbital binary eccentricity and frequency arXvi:1805.06194 eq 5.

        return ff0*((1.-ee0**2.)/(1.-ee**2.) * (ee/ee0)**(12./19.
                ) * (1.+121./304. * ee**2.)/(1.+121./304. * ee0**2.
                )**(870./2299))**(-3./2.)


pev = PetersEv()
