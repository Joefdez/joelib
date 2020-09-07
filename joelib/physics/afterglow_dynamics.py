from numpy import *
from joelib.constants.constants import cc, mp
from joelib.physics.afterglow_properties import normT, adabatic_index


def sound_speed(gam, beta):           # Sound speed prescription following LMR18

    TTs = normT(gam, beta)
    ada = adabatic_index(TTs)

    return cc*(ada*(ada-1.)*(gam-1.)/(1+ ada*(gam-1.)))**(1./2.)



def dthetadr(gamma, RR, theta, nn, aa):
    return 1./(RR*gamma**(1.+aa)*theta**(aa))



def dmdr(gamma, RR, thetaE, theta, nn, aa):
    t1 = (1./3.)*RR**2.*sin(theta)/(gamma**(1.+aa)*theta**(aa))          # First term: change in swept-up mass due to the change in solid angle
    t2 = (cos(thetaE)-cos(theta))*RR**2.          # Second term: change in swept-up mass due to radial expansion

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


# Functions from G&P 2012

def dMdr_GP12(Gamma, mm, RR, theta, nn):

    return 2.*pi*(theta*RR)**2.*mp*nn

def dgdR_GP12(Gamma, mm, RR, theta, nn):

    dmdr = dMdr_GP12(Gamma, mm, RR, theta, nn)

    return -1.*Gamma/(2.*mm) * dmdr




##########################################################################################################################################################################################
############################################################################### DYNAMICS ACCORDING TO ASAF PE'ER #########################################################################

def solver_collimated_shell(M0, gamma0, angExt0, RRs, nn, steps):

    """
    Solver for dynamics of relativistic, collimated shell.
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


def solver_expanding_shell(M0, gamma0, thetaE, theta0, RRs, nn, aa, steps, angExt0, cells, withSpread=True):


    #Solver for dynamics of laterally expaning shell.


    # First evolve swept up mass and theta, at fixed Lorentz factor
    #print shape(gamma0), shape(theta0)
    gammas, thetas, MMs, TTs, angExts = zeros(steps), zeros(steps), zeros(steps), zeros(steps), zeros(steps)
    gammas[0], thetas[0], angExts[0] = gamma0, theta0, angExt0
    MMs[0] = angExts[0]/3. * nn * mp * RRs[0]**3.


    for ii in range(1, steps):
        # First, calculate the evolution of Gamma, theta, and swept up mass
        RRn, RR = RRs[ii], RRs[ii-1]

        delR = RRn-RR

        theta, mms = thetas[ii-1], MMs[ii-1]
        gamma = gammas[ii-1]

        if (theta<pi) and (gamma<=1./sqrt(theta)):
            k1_theta  = delR*dthetadr(gamma, RR, theta, nn, aa)
            k1_mms    = delR*dmdr(gamma, RR, thetaE, theta, nn, aa)/cells
            k2_theta  = delR*dthetadr(gamma, RR+0.5*delR, theta + 0.5*k1_theta, nn, aa)
            k2_mms    = delR*dmdr(gamma, RR+0.5*delR, thetaE, theta + 0.5*k1_theta, nn, aa)/cells
            k3_theta  = delR*dthetadr(gamma, RR+0.5*delR, theta + 0.5*k2_theta, nn, aa)
            k3_mms    = delR*dmdr(gamma, RR+0.5*delR, thetaE, theta + 0.5*k2_theta, nn, aa)/cells
            k4_theta  = delR*dthetadr(gamma, RR + delR, theta + k3_theta, nn, aa)
            k4_mms    = delR*dmdr(gamma, RR+delR, thetaE, theta + k3_theta, nn, aa)/cells

            thetas[ii]   = theta + (1./6.) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
            MMs[ii]      = mms + (1./6.) * (k1_mms + 2 * k2_mms + 2 * k3_mms + k4_mms)


        else:   # If the shell is already spherical, stop considering lateral expansion in the swept-up mass
            MMs[ii] = mms + 2.*pi*(cos(thetaE)-cos(theta))*RR**2.*delR*nn*mp/cells
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

    if withSpread:
        betas = sqrt(1.-gammas**(-2.))
        dThetadr = concatenate([zeros(1), diff(thetas)/diff(RRs)])
        dR       = concatenate([zeros(1), diff(RRs)])
        integrand = 1./(cc*betas) * sqrt(1.+RRs**2.*dThetadr**2.) - 1./(cc)
        TTs[0] = RRs[0]/(cc*betas[0])* (sqrt(1.+RRs[0]**2.*dThetadr[0]**2.)) - RRs[0]/cc

    else:
        integrand = 1./(cc*gammas**2.*betas*(1.+betas))
        TTs[0] = RRs[0]/(cc*gammas[0]**2.*betas[0]*(1.+betas[0]))

    for ii in range(1,steps):
        TTs[ii] = trapz(integrand[0:ii+1], RRs[0:ii+1]) + TTs[0]

    # Just to finish off, calculate the solid angle extent of each ring

    angExts = 2.*pi*(1.-cos(thetas))/cells


    return gammas, betas, thetas, MMs, TTs, angExts    # Return 2*thetas, because theta is half the opening angle


def solver_GP12(M0, gamma0, thetaE, theta0, RRs, nn, aa, steps, angExt0, cells, withSpread=True):


    #Solver for dynamics of laterally expaning shell.


    # First evolve swept up mass and theta, at fixed Lorentz factor
    #print shape(gamma0), shape(theta0)
    gammas, thetas, MMs, TTs, angExts = zeros(steps), zeros(steps), zeros(steps), zeros(steps), zeros(steps)
    gammas[0], thetas[0], angExts[0] = gamma0, theta0, angExt0
    MMs[0] = angExts[0]/3. * nn * mp * RRs[0]**3.


    for ii in range(1, 5):
        # First, calculate the evolution of Gamma, theta, and swept up mass
        RRn, RR = RRs[ii], RRs[ii-1]

        delR = RRn-RR

        theta, mm = thetas[ii-1], MMs[ii-1]
        gamma = gammas[ii-1]

        if (theta<pi): # and (gamma<=1./sqrt(theta)):
            k1_theta  = delR*dthetadr(gamma, RR, theta, nn, aa)
            k1_mms    = delR*dMdr_GP12(gamma, mm, RR, theta, nn)/cells
            k1_gamma  = delR*dgdR_GP12(gamma, mm, RR, theta, nn)
            k2_theta  = delR*dthetadr(gamma  + 0.5*k1_gamma, RR+0.5*delR, theta + 0.5*k1_theta, nn, aa)
            k2_mms    = delR*dMdr_GP12(gamma + 0.5*k1_gamma, mm + 0.5*k1_mms, RR+0.5*delR, theta + 0.5*k1_theta, nn)/cells
            k2_gamma  = delR*dgdR_GP12(gamma + 0.5*k1_gamma, mm + 0.5*k1_mms, RR+0.5*delR, theta + 0.5*k1_theta, nn)
            k3_theta  = delR*dthetadr(gamma  + 0.5*k2_gamma, RR+0.5*delR, theta + 0.5*k2_theta, nn, aa)
            k3_mms    = delR*dMdr_GP12(gamma + 0.5*k2_gamma, mm + 0.5*k2_mms, RR+0.5*delR, theta + 0.5*k2_theta, nn)/cells
            k3_gamma  = delR*dgdR_GP12(gamma + 0.5*k2_gamma, mm + 0.5*k2_mms, RR+0.5*delR, theta + 0.5*k2_theta, nn)
            k4_theta  = delR*dthetadr(gamma  + k3_gamma, RR + delR, theta + k3_theta, nn, aa)
            k4_mms    = delR*dMdr_GP12(gamma + k3_gamma, mm + k3_mms, RR+delR, theta + k3_theta, nn)#/cells
            k4_gamma  = delR*dgdR_GP12(gamma + k3_gamma, mm + k3_mms, RR+delR, theta + k3_theta, nn)

            thetas[ii]   = theta + (1./6.) * (k1_theta + 2.*k2_theta + 2.*k3_theta + k4_theta)
            MMs[ii]      = mm    + (1./6.) * (k1_mms   + 2.*k2_mms   + 2.*k3_mms   + k4_mms)
            gammas[ii]   = gamma + (1./6.) * (k1_gamma + 2.*k2_gamma + 2.*k3_gamma + k4_gamma)

            print "Gam: ", k1_gamma, k2_gamma, k3_gamma, k4_gamma, "Tot: ",  (1./6.) * (k1_gamma + 2.*k2_gamma + 2.*k3_gamma + k4_gamma)
            print "theta: ", k1_theta, k2_theta, k3_theta, k4_theta, "Tot ", (1./6.) * (k1_theta + 2.*k2_theta + 2.*k3_theta + k4_theta)
            print "mm: ", k1_mms, k2_mms, k3_mms, k4_mms, "Tot ", (1./6.) * (k1_mms   + 2.*k2_mms   + 2.*k3_mms   + k4_mms)


        else:   # If the shell is already spherical, stop considering lateral expansion in the swept-up mass
            k1_gamma  = delR*dgdR_GP12(gamma, mm, RR, theta, nn)
            k1_mms    = delR*dMdr_GP12(gamma, mm, RR, theta, nn)/cells
            k2_gamma  = delR*dgdR_GP12(gamma + 0.5*k1_gamma, mm + 0.5*k1_mms, RR+0.5*delR, theta, nn)
            k2_mms    = delR*dMdr_GP12(gamma + 0.5*k1_gamma, mm + 0.5*k1_mms, RR+0.5*delR, theta, nn)/cells
            k3_gamma  = delR*dgdR_GP12(gamma + 0.5*k2_gamma, mm + 0.5*k2_mms, RR+0.5*delR, theta, nn)
            k3_mms    = delR*dMdr_GP12(gamma + 0.5*k3_gamma, mm + 0.5*k2_mms, RR+0.5*delR, theta, nn)/cells
            k4_gamma  = delR*dgdR_GP12(gamma + k3_gamma, mm + k3_mms, RR+delR, theta, nn)
            k4_mms    = delR*dMdr_GP12(gamma + k3_gamma, mm + k3_mms, RR+delR, theta, nn)/cells

            MMs[ii]      = mm   + (1./6.) * (k1_mms + 2 * k2_mms + 2 * k3_mms + k4_mms)
            gammas[ii]   = gamma + (1./6.) * (k1_gamma + 2 * k2_gamma + 2 * k3_gamma + k4_gamma)
            thetas[ii] = theta


    # Next calculate the on-axis time for a distant observer

    betas = sqrt(1.-gammas**(-2.))

    if withSpread:
        betas = sqrt(1.-gammas**(-2.))
        dThetadr = concatenate([zeros(1), diff(thetas)/diff(RRs)])
        dR       = concatenate([zeros(1), diff(RRs)])
        integrand = 1./(cc*betas) * sqrt(1.+RRs**2.*dThetadr**2.) - 1./(cc)
        TTs[0] = RRs[0]/(cc*betas[0])* (sqrt(1.+RRs[0]**2.*dThetadr[0]**2.)) - RRs[0]/cc

    else:
        integrand = 1./(cc*gammas**2.*betas*(1.+betas))
        TTs[0] = RRs[0]/(cc*gammas[0]**2.*betas[0]*(1.+betas[0]))

    for ii in range(1,steps):
        TTs[ii] = trapz(integrand[0:ii+1], RRs[0:ii+1]) + TTs[0]

    # Just to finish off, calculate the solid angle extent of each ring

    angExts = 2.*pi*(1.-cos(thetas))/cells


    return gammas, betas, thetas, MMs, TTs, angExts    # Return 2*thetas, because theta is half the opening angle





##########################################################################################################################################################################################
############################################################################### DYNAMICS LIKE BM #########################################################################

def BMsolver_collimated_shell(M0, gamma0, angExt0, RRs, nn, steps):

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


def BMsolver_expanding_shell(M0, gamma0, thetaE, theta0, RRs, nn, aa, steps, angExt0, cells):


    #Solver for dynamics of laterally expaning shell.
    #Lateral expansion implemented as in GP12 or LMR18


    # First evolve swept up mass and theta, at fixed Lorentz factor
    #print shape(gamma0), shape(theta0)
    gammas, thetas, MMs, TTs, angExts = zeros(steps), zeros(steps), zeros(steps), zeros(steps), zeros(steps)
    gammas[0], thetas[0], angExts[0] = gamma0, theta0, angExt0
    MMs[0] = angExts[0]/3. * nn * mp * RRs[0]**3.


    for ii in range(1, steps):
        # First, calculate the evolution of Gamma, theta, and swept up mass
        RRn, RR = RRs[ii], RRs[ii-1]

        delR = RRn-RR

        theta, mms = thetas[ii-1], MMs[ii-1]
        gamma = gammas[ii-1]

        if (theta<pi): #and (gamma<=4.):
            k1_theta  = delR*dthetadr(gamma, RR, theta, nn, aa)
            k1_mms    = delR*dmdr(gamma, RR, thetaE, theta, nn, aa)/cells
            k2_theta  = delR*dthetadr(gamma, RR+0.5*delR, theta + 0.5*k1_theta, nn, aa)
            k2_mms    = delR*dmdr(gamma, RR+0.5*delR, thetaE, theta + 0.5*k1_theta, nn, aa)/cells
            k3_theta  = delR*dthetadr(gamma, RR+0.5*delR, theta + 0.5*k2_theta, nn, aa)
            k3_mms    = delR*dmdr(gamma, RR+0.5*delR, thetaE, theta + 0.5*k2_theta, nn, aa)/cells
            k4_theta  = delR*dthetadr(gamma, RR + delR, theta + k3_theta, nn, aa)
            k4_mms    = delR*dmdr(gamma, RR+delR, thetaE, theta + k3_theta, nn, aa)/cells

            thetas[ii]   = theta + (1./6.) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
            MMs[ii]      = mms + (1./6.) * (k1_mms + 2 * k2_mms + 2 * k3_mms + k4_mms)


        else:   # If the shell is already spherical, stop considering lateral expansion in the swept-up mass
            MMs[ii] = mms + 2.*pi*(cos(thetaE)-cos(theta))*RR**2.*delR*nn*mp/cells
            thetas[ii] = theta

        delM = log10(MMs[ii]) - log10(mms)

        k1_gamma = delM*dgdm(M0, gamma, log10(mms))
        k2_gamma = delM*dgdm(M0, gamma + 0.5*k1_gamma, log10(mms)+0.5*delM)
        k3_gamma = delM*dgdm(M0, gamma + 0.5*k2_gamma, log10(mms)+0.5*delM)
        k4_gamma = delM*dgdm(M0, gamma + k3_gamma, log10(mms)+delM)

        #print k1_gamma, k2_gamma, k3_gamma, k4_gamma

        gammas[ii] = gamma + (1./6.) * (k1_gamma + 2 * k2_gamma + 2 * k3_gamma + k4_gamma)


    # Next calculate the on-axis time for a distant observer
    """
    betas = sqrt(1.-gammas**(-2.))
    integrand = 1./(cc*gammas**2.*betas*(1.+betas))
    TTs[0] = RRs[0]/(cc*gammas[0]**2.*betas[0]*(1.+betas[0]))
    #dtdr = dthetadr(gammas, RRs, thetas, nn, aa)
    #TTs[0] = RRs[0]*sqrt(1.+ (RRs[0]*dtdr[0])**2)/(cc*betas[0]) - RRs[0]/cc
    #integrand = sqrt(1.+ (RRs*dtdr)**2)/(cc*betas)
    for ii in range(1,steps):
        TTs[ii] = trapz(integrand[0:ii+1], RRs[0:ii+1]) + TTs[0]
    """
    betas = sqrt(1.-gammas**(-2.))
    dThetadr = concatenate([zeros(1), diff(thetas)/diff(RRs)])
    dR       = concatenate([zeros(1), diff(RRs)])
         #integrand1 = 1./(cc*Gams**2.*Betas*(1.+Betas))
    integrand = 1./(cc*betas) * sqrt(1.+RRs**2.*dThetadr**2.) - 1./(cc)

         #TTs_ne[0] = RRs[0]/(cc*Gams[0]**2.*Betas[0]*(1.+Betas[0]))
    TTs[0] = RRs[0]/(cc*betas[0])* (sqrt(1.+RRs[0]**2.*dThetadr[0]**2.)) - RRs[0]/cc
    for ii in range(1,steps):
             #TTs_ne[ii] = trapz(integrand1[0:ii+1], RRs[0:ii+1]) + TTs_ne[0]
             TTs[ii] = trapz(integrand[0:ii+1], RRs[0:ii+1]) + TTs[0]


    #TTs = TTs - RRs/cc



    # Just to finish off, calculate the solid angle extent of each ring

    angExts = 2.*pi*(1.-cos(thetas))/cells


    return gammas, betas, thetas, MMs, TTs, angExts    # Return 2*thetas, because theta is half the opening angle



############################################################################### DYNAMICS LIKE BM #########################################################################
##########################################################################################################################################################################################


"""
def solver_expanding_shell(M0, gamma0, thetaE, theta0, initJoAngle, RRs, nn, aa, steps, angExt0, cells):


    Solver for dynamics of laterally expaning shell.
    Lateral expansion implemented as in GP12 or LMR18


    # First evolve swept up mass and theta, at fixed Lorentz factor
    #print shape(gamma0), shape(theta0)
    gammas, thetas, MMs, TTs, angExts = zeros(steps), zeros(steps), zeros(steps), zeros(steps), zeros(steps)
    thetasOut = zeros(steps)
    gammas[0], thetas[0], thetasOut[0], angExts[0] = gamma0, initJoAngle, theta0, angExt0
    MMs[0] = angExts[0]/3. * nn * mp * RRs[0]**3.

    print(cos(thetaE)- cos(thetasOut[0]))/cells
    for ii in range(1, steps):
        # First, calculate the evolution of Gamma, theta, and swept up mass
        RRn, RR = RRs[ii], RRs[ii-1]

        delR = RRn-RR

        theta, mms = thetas[ii-1], MMs[ii-1]
        thetaOut = thetasOut[ii-1] + (theta-thetas[0])
        gamma = gammas[ii-1]


        print(thetaE, thetasOut[ii], theta, MMs[ii-1], MMs[ii])


        if theta<pi:
            k1_theta  = delR*dthetadr(gamma, RR, theta, nn, aa)
            k1_mms    = delR*dmdr(gamma, RR, thetaE, thetaOut, nn, aa)/cells
            k2_theta  = delR*dthetadr(gamma, RR+0.5*delR, theta + 0.5*k1_theta, nn, aa)
            k2_mms    = delR*dmdr(gamma, RR+0.5*delR, thetaE, thetaOut + 0.5*k1_theta, nn, aa)/cells
            k3_theta  = delR*dthetadr(gamma, RR+0.5*delR, theta + 0.5*k2_theta, nn, aa)
            k3_mms    = delR*dmdr(gamma, RR+0.5*delR, thetaE, thetaOut + 0.5*k2_theta, nn, aa)/cells
            k4_theta  = delR*dthetadr(gamma, RR + delR, theta + k3_theta, nn, aa)
            k4_mms    = delR*dmdr(gamma, RR+delR, thetaE, thetaOut + k3_theta, nn, aa)/cells

            thetas[ii]   = theta + (1./6.) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
            #thetasOut[ii] = thetaOut + (1./6.) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
            MMs[ii]      = mms + (1./6.) * (k1_mms + 2 * k2_mms + 2 * k3_mms + k4_mms)



        else:   # If the shell is already spherical, stop considering lateral expansion in the swept-up mass
            MMs[ii] = mms + 2.*pi*(cos(thetaE)-cos(thetaOut))*RR**2.*delR*nn*mp/cells
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
    #dtdr = dthetadr(gammas, RRs, thetas, nn, aa)
    #TTs[0] = RRs[0]*sqrt(1.+ (RRs[0]*dtdr[0])**2)/(cc*betas[0]) - RRs[0]/cc
    #integrand = sqrt(1.+ (RRs*dtdr)**2)/(cc*betas)
    for ii in range(1,steps):
        TTs[ii] = trapz(integrand[0:ii+1], RRs[0:ii+1]) + TTs[0]

    #TTs = TTs - RRs/cc


    # Just to finish off, calculate the solid angle extent of each ring

    angExts = 2.*pi*(1.-cos(thetas))/cells


    return gammas, betas, thetas, MMs, TTs, angExts    # Return 2*thetas, because theta is half the opening angle

"""


def obsTime_onAxis_integrated(RRs, Gams, Betas):

    """
    Very crude numerical integration to obtain the on-axis observer time
    """

    TTs = zeros(len(Betas))

    integrand = 1./(cc*Gams**2.*Betas*(1.+Betas))

    TTs[0] = RRs[0]/(cc*Gams[0]**2.*Betas[0]*(1.+Betas[0]))
    for ii in range(1,len(Betas)):
        TTs[ii] = trapz(integrand[0:ii+1], RRs[0:ii+1]) + TTs[0]


    return TTs


def obsTime_onAxis_LE_integrated(RRs, thetas, Gams, Betas):

    """
    Very crude numerical integration to obtain the on-axis observer time
    accounting for lateral expansion
    """

    TTs_ee = zeros(len(Betas))

    dthetadr = concatenate([zeros(1), diff(thetas)/diff(RRs)])
    dR       = concatenate([zeros(1), diff(RRs)])
    #integrand1 = 1./(cc*Gams**2.*Betas*(1.+Betas))
    integrand2 = 1./(cc*Betas) * sqrt(1.+RRs**2.*dthetadr**2.) - 1./(cc)

    #TTs_ne[0] = RRs[0]/(cc*Gams[0]**2.*Betas[0]*(1.+Betas[0]))
    TTs_ee[0] = RRs[0]/(cc*Betas[0])* (sqrt(1.+RRs[0]**2.*dthetadr[0]**2.)) - RRs[0]/cc
    for ii in range(1,len(Betas)):
        #TTs_ne[ii] = trapz(integrand1[0:ii+1], RRs[0:ii+1]) + TTs_ne[0]
        TTs_ee[ii] = trapz(integrand2[0:ii+1], RRs[0:ii+1]) + TTs_ee[0]


    return TTs_ee #TTs_ne, TTs_ee


def obsTime_offAxis_General_NEXP(RR, TT, theta):

    return TT + RR/cc * (1.-cos(theta))


def obsTime_offAxis_General_EXP(RRs, TTs, costhetas):

    delTTs = zeros(len(RRs))
    delTTs[0] =  RRs[0] * (1.-costhetas[0])
    for ii in range(1, len(RRs)):
        delTTs[ii] = trapz((1.-costhetas[0:ii+1]), RRs[0:ii+1]) + delTTs[0]

    return TTs + delTTs/cc
