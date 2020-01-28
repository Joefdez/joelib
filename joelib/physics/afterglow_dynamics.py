from numpy import *
from joelib.constants.constants import cc, mp
from afterglow_properties import normT, adabatic_index


def sound_speed(gam, beta):           # Sound speed prescription following LMR18

    TTs = normT(gam, beta)
    ada = adabatic_index(TTs)

    return cc*(ada*(ada-1.)*(gam-1.)/(1+ ada*(gam-1.)))**(1./2.)



def dthetadr(gamma, RR, theta, nn, aa):
    return 1./(RR*gamma**(1.+aa)*theta**(aa))



def dmdr(gamma, RR, theta, nn, aa):
    t1 = (1./3.)*RR**2.*sin(theta)/(gamma**(1.+aa)*theta**(aa))          # First term: change in swept-up mass due to the change in solid angle
    t2 = (1.-cos(theta))*RR**2.          # Second term: change in swept-up mass due to radial expansion

    return 2.*pi*nn*mp*(t1+t2)


def dgdm(M0, gam, mm):
    beta = sqrt(1-1./gam**2.)
    TT =   normT(gam, beta)
    ada =  adabatic_index(TT)
    #numerator = -4.*pi*jet.nn*mp*rr**2. * ( ada*(gam**2.-1)-(ada-1)*gam*beta**2  )
    #denominator = jet.EE/(jet.Gam0*cc**2.) + 4./3.*pi*jet.nn*mp*rr**3.*(2.*ada*gam-(ada-1)*(1.+gam**(-2)))
    numerator = -10.**mm*log(10)*( ada*(gam**2.-1)-(ada-1)*gam*beta**2  )
    denominator = M0 + 10.**mm*(2.*ada*gam-(ada-1)*(1.+gam**(-2)))

    #print denominator

    return numerator/denominator




def solver_collimated_shell(M0, gamma0, angExt0, RRs, nn, steps):

    """
    Solver for dynamics of relativistic, collimated shell.
    Lateral expansion implemented as in GP12 or LMR18
    """

    gammas, TTs,  = zeros(steps), zeros(steps)
    #print gamma0
    gammas[0] = gamma0
    MMs = angExt0/3. * nn * mp * RRs**3.

    for ii in range(1, steps):
        # First, calculate the evolution of Gamma, theta, and swept up mass
        delM = log10(MMs[ii]) - log10(MMs[ii-1])
        mms = MMs[ii-1]
        gamma = gammas[ii-1]

        k1_gamma = delM*dgdm(M0, gamma, log10(mms))
        k2_gamma = delM*dgdm(M0, gamma + 0.5*k1_gamma, log10(mms)+0.5*delM)
        k3_gamma = delM*dgdm(M0, gamma + 0.5*k2_gamma, log10(mms)+0.5*delM)
        k4_gamma = delM*dgdm(M0, gamma + k3_gamma, log10(mms)+delM)

        #print k1_gamma, k2_gamma, k3_gamma, k4_gamma

        gammas[ii] = gamma + (1./6.) * (k1_gamma + 2 * k2_gamma + 2 * k3_gamma + k4_gamma)


    # Next calculate the on-axis time for a distant observer
    betas = sqrt(1.-gammas**(-2.))
    integrand = 1./(cc*gammas**2.*betas*(1.+betas))
    TTs[0] = RRs[0]/(cc*gammas[0]**2.*betas[0]*(1.+betas[0]))
    for ii in range(1,steps):
        TTs[ii] = trapz(integrand[0:ii+1], RRs[0:ii+1]) + TTs[0]

    return gammas, betas, MMs, TTs


def solver_expanding_shell(M0, gamma0, theta0, RRs, nn, aa, steps):

    """
    Solver for dynamics of laterally expaning shell.
    Lateral expansion implemented as in GP12 or LMR18
    """

    # First evolve swept up mass and theta, at fixed Lorentz factor
    #print shape(gamma0), shape(theta0)
    gammas, thetas, MMs, TTs, angExts = zeros(steps), zeros(steps), zeros(steps), zeros(steps), zeros(steps)
    gammas[0], thetas[0], angExts[0] = gamma0, theta0, 2.*pi*(1.-cos(theta0))
    MMs[0] = angExts[0]/3. * nn * mp * RRs[0]**3.


    for ii in range(1, steps):
        # First, calculate the evolution of Gamma, theta, and swept up mass
        RRn, RR = RRs[ii], RRs[ii-1]

        delR = RRn-RR

        theta, mms = thetas[ii-1], MMs[ii-1]
        gamma = gammas[ii-1]

        if theta<=pi:


            k1_theta  = delR*dthetadr(gamma, RR, theta, nn, aa)
            k1_mms    = delR*dmdr(gamma, RR, theta, nn, aa)
            k2_theta  = delR*dthetadr(gamma, RR+0.5*delR, theta + 0.5*k1_theta, nn, aa)
            k2_mms    = delR*dmdr(gamma, RR+0.5*delR, theta + 0.5*k1_theta, nn, aa)
            k3_theta  = delR*dthetadr(gamma, RR+0.5*delR, theta + 0.5*k2_theta, nn, aa)
            k3_mms    = delR*dmdr(gamma, RR+0.5*delR, theta + 0.5*k2_theta, nn, aa)
            k4_theta  = delR*dthetadr(gamma, RR + delR, theta + k3_theta, nn, aa)
            k4_mms    = delR*dmdr(gamma, RR+delR, theta + k3_theta, nn, aa)

            thetas[ii]   = theta + (1./6.) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
            MMs[ii]      = mms + (1./6.) * (k1_mms + 2 * k2_mms + 2 * k3_mms + k4_mms)

        else:   # If the shell is already spherical, stop considering lateral expansion in the swept-up mass
            MMs[ii] = mms + 2.*pi*(1.-cos(theta))*RR**2.*delR*nn*mp
            thetas[ii] = theta

        delM = log10(MMs[ii]) - log10(mms)

        k1_gamma = delM*dgdm(M0, gamma, log10(mms))
        k2_gamma = delM*dgdm(M0, gamma + 0.5*k1_gamma, log10(mms)+0.5*delM)
        k3_gamma = delM*dgdm(M0, gamma + 0.5*k2_gamma, log10(mms)+0.5*delM)
        k4_gamma = delM*dgdm(M0, gamma + k3_gamma, log10(mms)+delM)

        #print k1_gamma, k2_gamma, k3_gamma, k4_gamma

        gammas[ii] = gamma + (1./6.) * (k1_gamma + 2 * k2_gamma + 2 * k3_gamma + k4_gamma)


    # Next calculate the on-axis time for a distant observer
    betas = sqrt(1.-gammas**(-2.))
    integrand = 1./(cc*gammas**2.*betas*(1.+betas))
    TTs[0] = RRs[0]/(cc*gammas[0]**2.*betas[0]*(1.+betas[0]))
    for ii in range(1,steps):
        TTs[ii] = trapz(integrand[0:ii+1], RRs[0:ii+1]) + TTs[0]

    # Just to finish off, calculate the solid angle extent of each ring

    angExts = 2.*pi*(1.-cos(thetas))


    return gammas, betas, thetas, MMs, TTs, angExts    # Return 2*thetas, because theta is half the opening angle


def obsTime_onAxis_integrated(RRs, Gams, Betas):

    "Very crude numerical integration to obtain the on-axis observer time"

    TTs = zeros(len(Betas))

    integrand = 1./(cc*Gams**2.*Betas*(1.+Betas))

    TTs[0] = RRs[0]/(cc*Gams[0]**2.*Betas[0]*(1.+Betas[0]))
    for ii in range(1,len(Betas)):
        TTs[ii] = trapz(integrand[0:ii+1], RRs[0:ii+1]) + TTs[0]


    return TTs


def obsTime_offAxis_General_NEXP(RR, TT, theta):

    return TT + RR/cc * (1.-cos(theta))


def obsTime_offAxis_General_EXP(RRs, TTs, cthetas):

    delTTs = zeros(len(RRs))
    delTTs[0] =  RRs[0]/(cc) * cthetas[0]
    for ii in range(1, len(RRs)):
        delTTs[ii] = trapz((1-cthetas[0:ii+1]), RRs[0:ii+1]) + delTTs[0]

    return TTs + delTTs[ii]/cc
