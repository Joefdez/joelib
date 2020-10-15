from numpy import *
from joelib.constants.constants import cc, mp, me, qe, sigT, sTd
from tqdm import *
from joelib.physics.afterglow_dynamics import obsTime_offAxis_General_NEXP, obsTime_offAxis_General_EXP
from joelib.physics.afterglow_properties import *
from scipy.interpolate import interp1d
from scipy.interpolate import griddata as gdd
from matplotlib.pylab import *
from scipy.ndimage import gaussian_filter

# Utilities and functions for calculating GRB observables, such as light curves for different components of the emission and synthetic images ##########


def obsangle(thetas, phis, alpha_obs):
        """
        Return the cosine of the observer angle for the different shockwave segments and and
        and observer at and angle alpha_obs with respect to the jet axis
        (contained in yz plane)
        """
        #u_obs_x, u_obs_y, u_obs_z = 0., sin(alpha_obs), cos(alpha_obs)
        u_obs_y, u_obs_z = sin(alpha_obs), cos(alpha_obs)

        #seg_x =
        seg_y = sin(thetas)*sin(phis)
        seg_z = cos(thetas)

        #return  arccos(u_obs_x*seg_x + u_obs_y*seg_y + u_obs_z*seg_z)
        return  u_obs_y*seg_y + u_obs_z*seg_z

def obsangle_cj(thetas, phis, alpha_obs):
        """
        Return the cosine of the observer angle for the different shockwave
        segments in the counter jet and observer at an angle alpha_obs with respect to the jet axis
        (contained in yz plane)
        """
        #u_obs_x, u_obs_y, u_obs_z = 0., sin(alpha_obs), cos(alpha_obs)
        u_obs_y, u_obs_z = sin(alpha_obs), cos(alpha_obs)

        #seg_x =
        seg_y = sin(pi-thetas)*sin(phis)
        seg_z = cos(pi-thetas)

        #return  arccos(u_obs_x*seg_x + u_obs_y*seg_y + u_obs_z*seg_z)
        return  u_obs_y*seg_y + u_obs_z*seg_z

def dopplerFactor(cosa, beta):
    """
    Calculate the doppler factors of the different jethead segments
    cosa -> cosine of observeration angle, obtained using obsangle
    """

    return (1.-beta)/(1.-beta*cosa)



def light_curve_peer_TH(jet, pp, alpha_obs, obsFreqs, DD, rangeType, timeD, Rb):


        # Takes top hat jet as input parameter!

        if rangeType=='range':

            tt0, ttf, num = timeD

            lt0 = log10(tt0*sTd) # Convert to seconds and then logspace
            ltf = log10(ttf*sTd) # Convert to seconds and then logspace
            tts = logspace(lt0, ltf+(ltf-lt0)/num, num) # Timeline on which the flux is evaluated.

        elif rangeType=='discrete':

            tts, num = timeD, len(timeD)


        # Takes top hat jet as input parameter!

        if type(obsFreqs)==float:
            obsFreqs  = array([obsFreqs])

        #calpha = obsangle(jet.cthetas, jet.cphis, alpha_obs)
        #alpha  = arccos(calpha)

        #calpha_cj = obsangle_cj(jet.cthetas, jet.cphis, alpha_obs)
        #alpha_cj  = arccos(calpha_cj)




#        if (ttf>max_Tobs or ttf>max_Tobs_cj):
#            print("ttf larger than maximum observable time. Adjusting value. ")
#            ttf = min(max_Tobs, max_Tobs_cj)

        lt0 = log10(tt0*sTd) # Convert to seconds and then logspace
        ltf = log10(ttf*sTd) # Convert to seconds and then logspace

        tts = logspace(lt0, ltf+(ltf-lt0)/num, num) # Timeline on which the flux is evaluated.


        light_curve      = zeros([len(obsFreqs), num])
        light_curve_RS   = zeros([len(obsFreqs), num])
        light_curve_CJ   = zeros([len(obsFreqs), num])


        for ii in tqdm(range(jet.ncells)):
        #for ii in range(jet.ncells):
            layer      = jet.layer[ii]
            #print(layer, type(layer))
            #theta_cell = jet.cthetas[layer-1]
            phi_cell   = jet.cphis[ii]
            #calpha, calpha_cj = obsangle(theta_cell, phi_cell, alpha_obs), obsangle_cj(theta_cell, phi_cell, alpha_obs)
            #alpha,  alpha_cj = arccos(calpha), arccos(calpha_cj)

            onAxisTint = interp1d(jet.RRs, jet.TTs)

            if jet.aa >= 0:
                # For laterally expanding shells
                #theta_cellR = ones([jet.steps])*jet.cthetas[0,layer-1] + 0.5*arcsin(sin(jet.thetas[:, layer-1])-sin(jet.initJoAngle))
                #cthetas_cell = jet.cthetas[:,layer-1]
                cthetas_cell = jet.cthetas[0,layer-1]
                calphaR, calphaR_cj = obsangle(cthetas_cell, phi_cell, alpha_obs), obsangle_cj(cthetas_cell, phi_cell, alpha_obs)
                #alphaR, alphaR_cj   = arccos(calphaR), arccos(calphaR_cj)
                #alphaRI, alphaR_cjI = interp1d(jet.RRs, alphaR), interp1d(jet.RRs, alphaR_cj)
                #calphaRI, calphaR_cjI = interp1d(jet.RRs, calphaR), interp1d(jet.RRs, calphaR_cj)
                #calphaR =
                ttobs = jet.TTs + jet.RRs/cc * (1.-calphaR)
                ttobs_cj = jet.TTs + jet.RRs/cc*(1.-calphaR_cj)
                #ttobs = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs, alphaR)
                #print(ttobs.min())/sTd
                #ttobs_cj = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs, calphaR_cj)

            else:
                # For collimated shells
                calpha, calpha_cj = obsangle(jet.cthetas[0,layer-1], phi_cell, alpha_obs), obsangle(jet.cthetas[0,layer-1], phi_cell, alpha_obs)
                alpha,  alpha_cj    = arccos(calpha), arccos(calpha_cj)
                ttobs = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs, alpha)
                ttobs_cj = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs, alpha_cj)



            #ttobs = obsTime_offAxis_UR(jet.RRs, jet.TTs, jet.Betas, alpha)

            filTM  = where(tts<=max(ttobs))[0]
            filTm  = where(tts[filTM]>=min(ttobs))[0]
            filTM_cj  = where(tts<=max(ttobs_cj))[0]
            filTm_cj  = where(tts[filTM_cj]>=min(ttobs_cj))[0]
            #print shape(filTM_cj[filTM_cj])
            #print shape(filTM_cj[filTM_cj])
            #print(len(tts[filT]))

            Rint = interp1d(ttobs, jet.RRs)
            Robs = Rint(tts[filTM][filTm])
            #Robs = Rint(tts)
            GamObs = jet.GamInt(Robs)
            BetaObs = sqrt(1.-GamObs**(-2.))
            #if jet.evolution == 'adiabatic':
            #    onAxisTobs = obsTime_onAxis_adiabatic(Robs, BetaObs)
            #elif jet.evolution == 'peer':
            #    onAxisTobs = obsTime_onAxis_integrated(Robs, GamObs, BetaObs)
            onAxisTobs = onAxisTint(Robs)
            thetaObs = jet.cthetasI[layer-1](Robs)
            calpha = obsangle(thetaObs, phi_cell, alpha_obs)
            #calpha = cos(alpha)
            #angExt = jet.angExtI(Robs)
            nE = jet.neI(Robs)

            Rint_cj = interp1d(ttobs_cj, jet.RRs)
            Robs_cj= Rint(tts[filTM_cj][filTm_cj])
            GamObs_cj = jet.GamInt(Robs_cj)
            BetaObs_cj = sqrt(1.-GamObs_cj**(-2.))
            onAxisTobs_cj = onAxisTint(Robs_cj)
            thetaObs_cj = jet.cthetasI[layer-1](Robs_cj)
            calpha_cj = obsangle_cj(thetaObs_cj, phi_cell, alpha_obs)

            #angExt_cj = jet.angExtI(Robs_cj)
            nE_cj = jet.neI(Robs_cj)

            #if jet.aa>=0:
            #    alpha# = , alpha_cj  = alphaRI(Robs), alphaR_cjI(Robs)
            #    calpha, calpha_cj = calphaRI(Robs), calphaR_cjI(Robs)
            #else:
            #    alpha, alpha_cj = ones(len(Robs))*alpha, ones(len(Robs))*alpha_cj
            #    calpha, calpha_cj = ones(len(Robs))*calpha, ones(len(Robs))*calpha_cj



            # Forward shock stuff, principal jet
            """
            Bfield = Bfield_modified(GamObs, BetaObs, jet.nn, jet.epB)
            gamMobs, nuMobs = minGam_modified(GamObs, jet.epE, jet.epB, jet.nn, pp, Bfield, jet.Xp)
            gamCobs, nuCobs = critGam_modified(GamObs, jet.epE, jet.epB, jet.nn, pp, Bfield, onAxisTobs)
            Fnuobs = fluxMax_modified(Robs, GamObs, nE, Bfield, jet.PhiP)
            """

            nuMobs = jet.nuMI(Robs)
            nuCobs = jet.nuCI(Robs)
            Fnuobs = jet.FnuMaxI(Robs)

            # Forward shock, counter-jet stuff
            """
            Bfield_cj = Bfield_modified(GamObs_cj, BetaObs_cj, jet.nn, jet.epB)
            gamMobs_cj, nuMobs_cj = minGam_modified(GamObs_cj, jet.epE, jet.epB, jet.nn, pp, Bfield_cj, jet.Xp)
            gamCobs_cj, nuCobs_cj = critGam_modified(GamObs_cj, jet.epE, jet.epB, jet.nn, pp, Bfield_cj, onAxisTobs_cj)
            Fnuobs_cj = fluxMax_modified(Robs_cj, GamObs_cj, nE_cj, Bfield_cj, jet.PhiP)
            """

            nuMobs_cj = jet.nuMI(Robs_cj)
            nuCobs_cj = jet.nuCI(Robs_cj)
            Fnuobs_cj = jet.FnuMaxI(Robs_cj)


            # Reverse shock stuff
            nuM_RS, nuC_RS, Fnu_RS = params_tt_RS(jet, onAxisTobs, Rb)


            dopFacs =  dopplerFactor(calpha, BetaObs)
            #afac = angExt/maximum(angExt*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))

            dopFacs_cj =  dopplerFactor(calpha_cj, BetaObs_cj)
            #afac_cj = angExt_cj/maximum(angExt_cj*ones(num)[filTM_cj][filTm_cj], 2.*pi*(1.-cos(1./GamObs_cj)))

            for freq in obsFreqs:
                fil1, fil2 = where(nuMobs<=nuCobs)[0], where(nuMobs>nuCobs)[0]
                fil3, fil4 = where(nuM_RS<=nuC_RS)[0], where(nuM_RS>nuC_RS)[0]
                fil5, fil6 = where(nuMobs_cj<=nuCobs_cj)[0], where(nuMobs_cj>nuCobs_cj)[0]


                freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                freqs_cj = freq/dopFacs_cj

                light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                        (GamObs[fil1]*(1.-BetaObs[fil1]*calpha))**(-3.)  * FluxNuSC_arr(jet.pp, nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))#*calpha
                #light_curve[obsFreqs==freq, :] = light_curve[obsFreqs==freq, :] + (
                #                        (GamObs*(1.-BetaObs*calpha))**(-3.)  * FluxNuSC_arr(jet.pp, nuMobs, nuCobs, Fnuobs, freqs))#*calpha
                if len(fil2[fil2])>0:
                    light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                                (GamObs[fil2]*(1.-BetaObs[fil2]*calpha))**(-3.)  * FluxNuSC_arr(jet.pp, nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))#*calpha
                #light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] + (
                #                        (GamObs[fil3]*(1.-BetaObs[fil3]*calpha[fil3]))**(-3.) * FluxNuSC_arr(jet.pp, nuM_RS[fil3], nuC_RS[fil3], Fnu_RS[fil3], freqs[fil3]))#*calpha
                #light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] + (
                #                                    afac[fil4] * dopFacs[fil4]**3. * FluxNuFC_arr(jet, nuM_RS[fil4], nuC_RS[fil4], Fnu_RS[fil4], freqs[fil4]))*calpha
                light_curve_CJ[obsFreqs==freq, filTM_cj[filTm_cj][fil5]] = light_curve_CJ[obsFreqs==freq, filTM_cj[filTm_cj][fil5]] + (
                                        (GamObs_cj[fil5]*(1.-BetaObs_cj[fil5]*calpha_cj))**(-3.) * FluxNuSC_arr(jet.pp, nuMobs_cj[fil5], nuCobs_cj[fil5], Fnuobs_cj[fil5], freqs_cj[fil5]))#*calpha
                if len(fil6[fil6])>0:
                    light_curve_CJ[obsFreqs==freq, filTM_cj[filTm_cj][fil6]] = light_curve_CJ[obsFreqs==freq, filTM_cj[filTm_cj][fil6]] + (
                                        (GamObs_cj[fil6]*(1.-BetaObs_cj[fil6]*calpha_cj))**(-3.) * FluxNuSC_arr(jet.pp, nuMobs_cj[fil6], nuCobs_cj[fil6], Fnuobs_cj[fil6], freqs_cj[fil6]))#*calpha


        #return tts, 2.*light_curve, 2.*light_curve_RS
        return tts, light_curve/(DD**2.), light_curve_RS/(DD**2.), light_curve_CJ/(DD**2.)




def light_curve_peer_SJ(jet, pp, alpha_obs, obsFreqs, DD, rangeType, timeD, Rb):

        # Takes top hat jet as input parameter!

        if rangeType=='range':

            tt0, ttf, num = timeD

            lt0 = log10(tt0*sTd) # Convert to seconds and then logspace
            ltf = log10(ttf*sTd) # Convert to seconds and then logspace
            tts = logspace(lt0, ltf+(ltf-lt0)/num, num) # Timeline on which the flux is evaluated.

        elif rangeType=='discrete':

            tts, num = timeD*sTd, len(timeD)


        if type(obsFreqs)!=ndarray:
            obsFreqs  = array([obsFreqs])

        #calpha = obsangle(jet.cthetas, jet.cphis, alpha_obs)
        #alpha  = arccos(calpha)

        #calpha_cj = obsangle_cj(jet.cthetas, jet.cphis, alpha_obs)
        #alpha_cj  = arccos(calpha_cj)




#        if (ttf>max_Tobs or ttf>max_Tobs_cj):
#            print "ttf larger than maximum observable time. Adjusting value. "
#            ttf = min(max_Tobs, max_Tobs_cj)




        light_curve      = zeros([len(obsFreqs), num])
        light_curve_RS   = zeros([len(obsFreqs), num])
        light_curve_CJ   = zeros([len(obsFreqs), num])
        #f, a = subplots()
        #f2, a2 = subplots()

        #for ii in tqdm(range(jet.ncells)):
        for ii in range(jet.ncells):
            layer      = jet.layer[ii]
            if jet.cell_Gam0s[layer-1] <= 1.+1e-5:
                continue
            #print(layer, type(layer))
            #theta_cell = jet.cthetas[layer-1]
            phi_cell   = jet.cphis[ii]
            #calpha, calpha_cj = obsangle(theta_cell, phi_cell, alpha_obs), obsangle_cj(theta_cell, phi_cell, alpha_obs)
            #alpha,  alpha_cj = arccos(calpha), arccos(calpha_cj)

            #print(layer-1)
            GamInt = jet.GamInt[layer-1]
            onAxisTint  = jet.TTInt[layer-1]
            """
            if jet.aa >= 0:
                # For laterally expanding shells
                #theta_cellR = ones([jet.steps])*jet.cthetas[0,layer-1] + 0.5*arcsin(sin(jet.thetas[:, layer-1])-sin(jet.initJoAngle))
                theta_cellR = jet.cthetas[:,layer-1]
                calphaR, calphaR_cj = obsangle(theta_cellR, phi_cell, alpha_obs), obsangle_cj(theta_cellR, phi_cell, alpha_obs)
                #alphaR, alphaR_cj   = arccos(calphaR), arccos(calphaR_cj)
                #alphaRI, alphaR_cjI = interp1d(jet.RRs, alphaR), interp1d(jet.RRs, alphaR_cj)
                #ttobs = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs[:,layer-1], calphaR)
                #ttobs_cj = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs[:,layer-1], calphaR_cj)

                #cthetas_cell = jet.cthetas[:,layer-1]
                cthetas_cell = jet.cthetas0[layer-1]
                ttobs = jet.TTs[:, layer-1] + jet.RRs/cc * (1.-calphaR)
                ttobs_cj = jet.TTs[:,layer-1] + jet.RRs/cc* (1.-calphaR_cj)

            else:
                # For collimated shells
                calpha, calpha_cj = obsangle(jet.cthetas0[layer-1], phi_cell, alpha_obs), obsangle_cj(jet.cthetas0[layer-1], phi_cell, alpha_obs)
                alpha,  alpha_cj    = arccos(calpha), arccos(calpha_cj)
                ttobs = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs[:,layer-1], alpha)
                ttobs_cj = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs[:,layer-1], alpha_cj)
            """

            theta_cellR = jet.cthetas0[layer-1]
            calphaR, calphaR_cj = obsangle(theta_cellR, phi_cell, alpha_obs), obsangle_cj(theta_cellR, phi_cell, alpha_obs)
            ttobs = jet.TTs[:, layer-1] + jet.RRs/cc * (1.-calphaR)
            #ttobs_cj = jet.TTs[:,layer-1] + jet.RRs/cc* (1.-calphaR_cj)


            #ttobs = obsTime_offAxis_UR(jet.RRs, jet.TTs, jet.Betas, alpha)

            filTM  = where(tts<=max(ttobs))[0]
            filTm  = where(tts[filTM]>=min(ttobs))[0]
            #filTM_cj  = where(tts<=max(ttobs_cj))[0]
            #filTm_cj  = where(tts[filTM_cj]>=min(ttobs_cj))[0]
            #print shape(filTM_cj[filTM_cj])
            #print shape(filTM_cj[filTM_cj])
            #print(len(tts[filT]))

            Rint =  interp1d(ttobs, jet.RRs)
            Robs =  Rint(tts[filTM][filTm])
            GamObs = GamInt(Robs)
            BetaObs = sqrt(1.-GamObs**(-2.))
            #if jet.evolution == 'adiabatic':
            #    onAxisTobs = obsTime_onAxis_adiabatic(Robs, BetaObs)
            #elif jet.evolution == 'peer':
            #    onAxisTobs = obsTime_onAxis_integrated(Robs, GamObs, BetaObs)
            #onAxisTobs = onAxisTint(Robs)
            #thetaObs = jet.cthetasI[layer-1](Robs)
            #thetaObs = jet.cthetas0[layer-1]
            calpha = calphaR # obsangle(thetaObs, phi_cell, alpha_obs)
            nE = jet.neI[layer-1](Robs)
            #angExt = jet.angExtI[layer-1](Robs)


            #Rint_cj = interp1d(ttobs_cj, jet.RRs)
            #Robs_cj= Rint(tts[filTM_cj][filTm_cj])
            #GamObs_cj = jet.GamInt[layer-1](Robs_cj)
            #BetaObs_cj = sqrt(1.-GamObs_cj**(-2.))
            #onAxisTobs_cj = onAxisTint(Robs_cj)
            #thetaObs_cj = jet.cthetasI[layer-1](Robs_cj)
            #calpha_cj = obsangle_cj(thetaObs_cj, phi_cell, alpha_obs)
            #nE_cj = jet.neI[layer-1](Robs_cj)
            #angExt_cj = jet.angExtI[layer-1](Robs_cj)
            #if jet.aa>=0:
            #    alpha, alpha_cj  = alphaRI(Robs), alphaR_cjI(Robs)
            #    calpha, calpha_cj = cos(alpha), cos(alpha_cj)
            #else:
                #print(ii, layer-1, alpha*180/pi, shape(filTM[filTM]), shape(filTM[filTM][filTm]), onAxisTobs[0]/sTd, onAxisTobs[-1]/sTd)
            #    alpha, alpha_cj = ones(len(Robs))*alpha, ones(len(Robs))*alpha_cj
            #    calpha, calpha_cj = ones(len(Robs))*calpha, ones(len(Robs))*calpha_cj




            # Forward shock stuff, principal jet
            """
            Bfield = Bfield_modified(GamObs, BetaObs, jet.nn, jet.epB)
            gamMobs, nuMobs = minGam_modified(GamObs, jet.epE, jet.epB, jet.nn, pp, Bfield, jet.Xp)
            gamCobs, nuCobs = critGam_modified(GamObs, jet.epE, jet.epB, jet.nn, pp, Bfield, onAxisTobs)
            nuMobs, nuCobs = nuMobs*(1.-BetaObs)/(1.-BetaObs*calpha), nuCobs*(1.-BetaObs)/(1.-BetaObs*calpha)
            Fnuobs = fluxMax_modified(Robs, GamObs, nE, Bfield, jet.PhiP)
            a.loglog(tts[filTM][filTm]/sTd, nuMobs)
            a2.loglog(tts[filTM][filTm], GamObs)
            """
            nuMobs = jet.nuMI[layer-1](Robs)
            nuCobs = jet.nuCI[layer-1](Robs)
            Fnuobs = jet.FnuMax[layer-1](Robs)

            #nuMobs_cj = jet.nuMI[layer-1](Robs_cj)
            #nuCobs_cj = jet.nuCI[layer-1](Robs_cj)
            #Fnuobs_cj = jet.FnuMax[layer-1](Robs_cj)

            # Forward shock, counter-jet stuff
            #Bfield_cj = Bfield_modified(GamObs_cj, BetaObs_cj, jet.nn, jet.epB)
            #gamMobs_cj, nuMobs_cj = minGam_modified(GamObs_cj, jet.epE, jet.epB, jet.nn, pp, Bfield_cj, jet.Xp)
            #gamCobs_cj, nuCobs_cj = critGam_modified(GamObs_cj, jet.epE, jet.epB, jet.nn, pp, Bfield_cj, onAxisTobs_cj)
            #Fnuobs_cj = fluxMax_modified(Robs_cj, GamObs_cj, nE_cj, Bfield_cj, jet.PhiP)


            # Reverse shock stuff
            #nuM_RS, nuC_RS, Fnu_RS = params_tt_RS_SJ(jet, onAxisTobs, layer-1, Rb)


            dopFacs =  dopplerFactor(calpha, BetaObs)
            #afac = angExt/maximum(angExt*ones(num), 2.*pi*(1.-cos(1./GamObs)))
            #afac = maximum(ones(num), thetaObs**2*GamObs**2)
            #dopFacs_cj =  dopplerFactor(calpha_cj, BetaObs_cj)
            #afac_cj = angExt_cj/maximum(angExt_cj*ones(num)[filTM_cj][filTm_cj], 2.*pi*(1.-cos(1./GamObs_cj)))

            for freq in obsFreqs:
                fil1, fil2 = where(nuMobs<=nuCobs)[0], where(nuMobs>nuCobs)[0]
                #fil3, fil4 = where(nuM_RS<=nuC_RS)[0], where(nuM_RS>nuC_RS)[0]
                #fil5, fil6 = where(nuMobs_cj<=nuCobs_cj)[0], where(nuMobs_cj>nuCobs_cj)[0]


                freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                #freqs_cj = freq/dopFacs_cj

                #freqs = freq*ones(len(dopFacs))

                light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                        (GamObs[fil1]*(1.-BetaObs[fil1]*calpha))**(-3.) * FluxNuSC_arr(jet.pp, nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))
                #if len(fil2[fil2])>0:
                #    light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                #                        (GamObs[fil2]*(1.-BetaObs[fil2]*calpha[fil2]))**(-3.) * FluxNuFC_arr(jet.pp, nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))#*calpha

                #light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] + (
                #                        (GamObs[fil3]*(1.-BetaObs[fil3]*calpha))**(-3.) * FluxNuSC_arr(jet.pp, nuM_RS[fil3], nuC_RS[fil3], Fnu_RS[fil3], freqs[fil3]))#*calpha
                #light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] + (
                #                                    afac[fil4] * dopFacs[fil4]**3. * FluxNuFC_arr(jet, nuM_RS[fil4], nuC_RS[fil4], Fnu_RS[fil4], freqs[fil4]))*calpha
                #light_curve_CJ[obsFreqs==freq, filTM_cj[filTm_cj][fil5]] = light_curve_CJ[obsFreqs==freq, filTM_cj[filTm_cj][fil5]] + (
                #                        (GamObs_cj[fil5]*(1.-BetaObs_cj[fil5]*calpha_cj))**(-3.) * FluxNuSC_arr(jet.pp, nuMobs_cj[fil5], nuCobs_cj[fil5], Fnuobs_cj[fil5], freqs_cj[fil5]))#*calpha



        #return tts, 2.*light_curve, 2.*light_curve_RS
        return tts, light_curve/(DD**2.), light_curve_RS/(DD**2.), light_curve_CJ/(DD**2.)




def flux_at_time_SJ(jet, alpha_obs, tt_obs, freq, DD):


        ncells = jet.ncells

        #calpha = zeros([2*jet.ncells])
        #alpha  = zeros([2*jet.ncells])


        TTs, RRs, Gams= zeros(2*ncells), zeros(2*ncells), zeros(2*ncells)
        thetas, thetas0 = zeros(2*ncells), zeros(ncells)
        #nuMs, nuCs, fluxes    = zeros(2.*self.ncells), zeros(2.*self.ncells), zeros(2.*self.ncells)
        #fluxes = zeros(2*ncells)
        calphas = zeros(2*ncells)
        nE = zeros(2*ncells)



        for ii in tqdm(range(jet.ncells)):


            layer      = jet.layer[ii]
            phi_cell   = jet.cphis[ii]

            #onAxisTint = interp1d(jet.RRs, jet.TTs[:,layer-1])

            """
            if jet.aa >= 0:
                #cthetas_cell = jet.cthetas[:,layer-1]
                calphaR, calphaR_cj = obsangle(jet.cthetas[:,layer-1], phi_cell, alpha_obs), obsangle_cj(jet.cthetas[:,layer-1], phi_cell, alpha_obs)
                ttobs = jet.TTs[:,layer-1] + jet.RRs/cc * (1.-calphaR)
                #ttobs_cj = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs[:,layer-1], calphaR_cj)
                ttobs_cj = jet.TTs[:,layer-1] + jet.RRs/cc * (1.-calphaR_cj)
            else:
                calpha, calpha_cj = obsangle(jet.cthetas[:,layer-1], phi_cell, alpha_obs), obsangle_cj(jet.cthetas[:,layer-1], phi_cell, alpha_obs)
                alpha,  alpha_cj    = arccos(calpha), arccos(calpha_cj)
                ttobs = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs[:,layer-1], alpha)
                ttobs_cj = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs[:,layer-1], alpha_cj)
            """

            calphaR, calphaR_cj = obsangle(jet.cthetas[:,layer-1], phi_cell, alpha_obs), obsangle_cj(jet.cthetas[:,layer-1], phi_cell, alpha_obs)
            ttobs = jet.TTs[:,layer-1] + jet.RRs/cc * (1.-calphaR)
            #ttobs_cj = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs[:,layer-1], calphaR_cj)
            ttobs_cj = jet.TTs[:,layer-1] + jet.RRs/cc * (1.-calphaR_cj)

            Rint = interp1d(ttobs, jet.RRs)
            Rint_cj = interp1d(ttobs_cj, jet.RRs)
            Robs, Robs_cj = Rint(tt_obs), Rint_cj(tt_obs)
            RRs[ii], RRs[ii+ncells] = Robs, Robs_cj
            Gams[ii], Gams[ii+ncells] = jet.GamInt[layer-1](Robs), jet.GamInt[layer-1](Robs_cj)
            TTs[ii], TTs[ii+ncells] = jet.TTInt[layer-1](Robs), jet.TTInt[layer-1](Robs_cj)
            thetas[ii], thetas[ii+ncells] = jet.cthetasI[layer-1](Robs), jet.cthetasI[layer-1](Robs_cj)
            thetas0[ii] = jet.cthetas0[layer-1]
            nE[ii], nE[ii+ncells] = jet.neI[layer-1](Robs), jet.neI[layer-1](Robs_cj)


        Betas = sqrt(1.-Gams**(-2))
        calphas[:ncells], calphas[ncells:] = obsangle(thetas[:ncells], jet.cphis, alpha_obs), obsangle_cj(thetas[ncells:], jet.cphis, alpha_obs)
        #alphas = arccos(calphas)


        Bfield = Bfield_modified(Gams, Betas, jet.nn, jet.epB)
        gamMobs, nuMobs = minGam_modified(Gams, jet.epE, jet.epB, jet.nn, jet.pp, Bfield, jet.Xp)
        gamCobs, nuCobs = critGam_modified(Gams, jet.epE, jet.epB, jet.nn, jet.pp, Bfield, TTs)
        Fnuobs = fluxMax_modified(Robs, Gams, nE, Bfield, jet.PhiP)


        dopFacs= dopplerFactor(calphas, Betas)
        obsFreqs = freq/dopFacs
        #print(shape(nuMobs), shape(nuCobs), shape(Fnuobs), shape(obsFreqs))
        fluxes = 1./(DD**2.) * (Gams*(1.-Betas*calphas))**(-3.) * FluxNuSC_arr(jet.pp, nuMobs, nuCobs, Fnuobs, obsFreqs)



        return fluxes, thetas, thetas0, calphas, Gams



def skymapTH(jet, alpha_obs, tt_obs, freq):


    ncells = jet.ncells

    #calpha = zeros([2*jet.ncells])
    #alpha  = zeros([2*jet.ncells])


    TTs, RRs, Gams= zeros(2*ncells), zeros(2*ncells), zeros(2*ncells)
    thetas = zeros(2*ncells)
    #nuMs, nuCs, fluxes    = zeros(2.*self.ncells), zeros(2.*self.ncells), zeros(2.*self.ncells)
    #fluxes = zeros(2*ncells)
    calphas = zeros(2*ncells)
    im_xxs, im_yys = zeros(2*ncells), zeros(2*ncells)
    nE = zeros(2*ncells)

    if velocity:
        velX = zeros(2*ncells)
        velY = zeros(2*ncells)


    for ii in tqdm(range(jet.ncells)):


        layer      = jet.layer[ii]
        phi_cell   = jet.cphis[ii]


        calphaR, calphaR_cj = obsangle(jet.cthetas[:,layer-1], phi_cell, alpha_obs), obsangle_cj(jet.cthetas[:,layer-1], phi_cell, alpha_obs)
        ttobs = jet.TTs + jet.RRs/cc * (1.-calphaR)
        #ttobs_cj = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs[:,layer-1], calphaR_cj)
        ttobs_cj = jet.TTs[:,layer-1] + jet.RRs/cc * (1.-calphaR_cj)

        Rint = interp1d(ttobs, jet.RRs)
        Rint_cj = interp1d(ttobs_cj, jet.RRs)
        Robs, Robs_cj = Rint(tt_obs), Rint_cj(tt_obs)
        RRs[ii], RRs[ii+ncells] = Robs, Robs_cj
        Gams[ii], Gams[ii+ncells] = jet.GamInt(Robs), jet.GamInt(Robs_cj)
        TTs[ii], TTs[ii+ncells] = jet.TTInt(Robs), jet.TTInt(Robs_cj)
        thetas[ii], thetas[ii+ncells] = jet.cthetasI[layer-1](Robs), jet.cthetasI[layer-1](Robs_cj)
        nE[ii], nE[ii+ncells] = jet.neI[layer-1](Robs), jet.neI[layer-1](Robs_cj)


    Betas = sqrt(1.-Gams**(-2))
    calphas[:ncells], calphas[ncells:] = obsangle(thetas[:ncells], jet.cphis, alpha_obs), obsangle_cj(thetas[ncells:], jet.cphis, alpha_obs)
    #alphas = arccos(calphas)


    # Principal jet
    im_xxs[:ncells] = -1.*cos(alpha_obs)*sin(thetas[:ncells])*sin(jet.cphis) + sin(alpha_obs)*cos(thetas[:ncells])
    im_yys[:ncells] = sin(thetas[:ncells])*cos(jet.cphis)
    # Counter jet
    im_xxs[ncells:] = -1.*cos(alpha_obs)*sin(pi-thetas[ncells:])*sin(jet.cphis) + sin(alpha_obs)*cos(pi-thetas[ncells:])
    im_yys[ncells:] = sin(pi-thetas[ncells:])*cos(jet.cphis)

    Bfield = Bfield_modified(Gams, Betas, jet.nn, jet.epB)
    gamMobs, nuMobs = minGam_modified(Gams, jet.epE, jet.epB, jet.nn, jet.pp, Bfield, jet.Xp)
    gamCobs, nuCobs = critGam_modified(Gams, jet.epE, jet.epB, jet.nn, jet.pp, Bfield, TTs)
    Fnuobs = fluxMax_modified(Robs, Gams, nE, Bfield, jet.PhiP)


    dopFacs= dopplerFactor(calphas, Betas)
    obsFreqs = freq/dopFacs
    #print(shape(nuMobs), shape(nuCobs), shape(Fnuobs), shape(obsFreqs))
    fluxes = 1./(abs(calphas)*RRs**2.) * (Gams*(1.-Betas*calphas))**(-3.) * FluxNuSC_arr(jet.pp, nuMobs, nuCobs, Fnuobs, obsFreqs)

    return fluxes, RRs*im_xxs, RRs*im_yys, RRs, Gams, calphas, TTs



def skymapSJ(jet, alpha_obs, tt_obs, freq, velocity=False):

    ncells = jet.ncells

    #calpha = zeros([2*jet.ncells])
    #alpha  = zeros([2*jet.ncells])


    TTs, RRs, Gams= zeros(2*ncells), zeros(2*ncells), zeros(2*ncells)
    thetas, calphasR = zeros(2*ncells), zeros(2*ncells)
    #nuMs, nuCs, fluxes    = zeros(2.*self.ncells), zeros(2.*self.ncells), zeros(2.*self.ncells)
    #fluxes = zeros(2*ncells)
    calphas = zeros(2*ncells)
    im_xxs, im_yys = zeros(2*ncells), zeros(2*ncells)
    nE = zeros(2*ncells)

    if velocity:
        velX = zeros(2*ncells)
        velY = zeros(2*ncells)


    for ii in tqdm(range(jet.ncells)):


        layer      = jet.layer[ii]
        phi_cell   = jet.cphis[ii]

        #onAxisTint = interp1d(jet.RRs, jet.TTs[:,layer-1])

        """
        if jet.aa >= 0:
            #cthetas_cell = jet.cthetas[:,layer-1]
            calphaR, calphaR_cj = obsangle(jet.cthetas[:,layer-1], phi_cell, alpha_obs), obsangle_cj(jet.cthetas[:,layer-1], phi_cell, alpha_obs)
            ttobs = jet.TTs[:,layer-1] + jet.RRs/cc * (1.-calphaR)
            #ttobs_cj = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs[:,layer-1], calphaR_cj)
            ttobs_cj = jet.TTs[:,layer-1] + jet.RRs/cc * (1.-calphaR_cj)
        else:
            calpha, calpha_cj = obsangle(jet.cthetas[:,layer-1], phi_cell, alpha_obs), obsangle_cj(jet.cthetas[:,layer-1], phi_cell, alpha_obs)
            alpha,  alpha_cj    = arccos(calpha), arccos(calpha_cj)
            ttobs = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs[:,layer-1], alpha)
            ttobs_cj = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs[:,layer-1], alpha_cj)
        """

        #calphaR, calphaR_cj = obsangle(jet.cthetas[:,layer-1], phi_cell, alpha_obs), obsangle_cj(jet.cthetas[:,layer-1], phi_cell, alpha_obs)
        calphasR[ii], calphasR[ii+ncells] = obsangle(jet.cthetas0[layer-1], phi_cell, alpha_obs), obsangle_cj(jet.cthetas0[layer-1], phi_cell, alpha_obs)
        ttobs = jet.TTs[:,layer-1] + jet.RRs/cc * (1.-calphasR[ii])
        #ttobs_cj = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs[:,layer-1], calphaR_cj)
        ttobs_cj = jet.TTs[:,layer-1] + jet.RRs/cc * (1.-calphasR[ii+ncells])

        Rint = interp1d(ttobs, jet.RRs)
        Rint_cj = interp1d(ttobs_cj, jet.RRs)
        Robs, Robs_cj = Rint(tt_obs), Rint_cj(tt_obs)
        RRs[ii], RRs[ii+ncells] = Robs, Robs_cj
        Gams[ii], Gams[ii+ncells] = jet.GamInt[layer-1](Robs), jet.GamInt[layer-1](Robs_cj)
        TTs[ii], TTs[ii+ncells] = jet.TTInt[layer-1](Robs), jet.TTInt[layer-1](Robs_cj)
        thetas[ii], thetas[ii+ncells] = jet.cthetasI[layer-1](Robs), jet.cthetasI[layer-1](Robs_cj)
        nE[ii], nE[ii+ncells] = jet.neI[layer-1](Robs), jet.neI[layer-1](Robs_cj)


    Betas = sqrt(1.-Gams**(-2))
    calphas[:ncells], calphas[ncells:] = obsangle(thetas[:ncells], jet.cphis, alpha_obs), obsangle_cj(thetas[ncells:], jet.cphis, alpha_obs)
    #alphas = arccos(calphas)


    # Principal jet
    im_xxs[:ncells] = -1.*cos(alpha_obs)*sin(thetas[:ncells])*sin(jet.cphis) + sin(alpha_obs)*cos(thetas[:ncells])
    im_yys[:ncells] = sin(thetas[:ncells])*cos(jet.cphis)
    # Counter jet
    im_xxs[ncells:] = -1.*cos(alpha_obs)*sin(pi-thetas[ncells:])*sin(jet.cphis) + sin(alpha_obs)*cos(pi-thetas[ncells:])
    im_yys[ncells:] = sin(pi-thetas[ncells:])*cos(jet.cphis)

    Bfield = Bfield_modified(Gams, Betas, jet.nn, jet.epB)
    gamMobs, nuMobs = minGam_modified(Gams, jet.epE, jet.epB, jet.nn, jet.pp, Bfield, jet.Xp)
    gamCobs, nuCobs = critGam_modified(Gams, jet.epE, jet.epB, jet.nn, jet.pp, Bfield, TTs)
    Fnuobs = fluxMax_modified(Robs, Gams, nE, Bfield, jet.PhiP)


    #dopFacs= dopplerFactor(calphas, Betas)
    dopFacs = dopplerFactor(calphasR, Betas)
    obsFreqs = freq/dopFacs
    #print(shape(nuMobs), shape(nuCobs), shape(Fnuobs), shape(obsFreqs))
    fluxes = 1./(abs(calphasR)*RRs**2.) * (Gams*(1.-Betas*calphasR))**(-3.) * FluxNuSC_arr(jet.pp, nuMobs, nuCobs, Fnuobs, obsFreqs)

    return fluxes, RRs*im_xxs, RRs*im_yys, RRs, Gams, calphas, TTs



def skyMap_to_Grid(fluxes, xxs, yys, nx, ny=1, fac=1, scale=False, inter='linear'):

    # Function maps the coordinate output of the skymap functions to a grid.
    # Basically a wrapper for gdd and mgrid


    if (ny==1 and not scale):
            ny = nx
    elif (scale):
        dX = xxs.max()-xxs.min()
        dY = yys.max()-yys.min()
        fac = max(dX,dY)/min(dX,dY)
        print(fac)
        if(dY>=dX):
            ny = round(fac*nx)
        else:
            ny = nx
            nx = round(fac*nx)

    else:
        pass

    nx = complex(0,nx)
    ny = complex(0,ny)


    grid_x, grid_y = mgrid[xxs.min():xxs.max():nx, yys.min():yys.max():ny]
    image = gdd(array([xxs,yys]).T*fac, fluxes,
                    (grid_x*fac, grid_y*fac), method=inter, fill_value=0)

    # RETURNS ARRAY WITH NX ROWS AND NY COLUMS (i.e. each row, which represents a horizontal position in x has ny divisions)

    return grid_x[:,0], grid_y[0,:], image


def lateral_distributions(image, collapse_axis):

    if collapse_axis == 'x':
        points = image.shape[1]
        latAvDist  = array([image[:,ii].mean() for ii in range(points)])
        latMaxDist = array([image[:,ii].max() for ii in range(points)])

    elif collapse_axis == 'y':
        points = image.shape[0]
        latAvDist  = array([image[ii,:].mean() for ii in range(points)])
        latMaxDist = array([image[ii,:].max() for ii in range(points)])

    return latAvDist, latMaxDist


def image_slice(fluxes, xxs, yys, position, nn=50, axis='y', inter='linear', fac=1):

    nn = complex(0,nn)
    if axis=='y':
        grid_x, grid_y = mgrid[position:position:1j, yys.min():yys.max():nn]
    else:
        grid_x, grid_y = mgrid[xxs.min():xxs.max():nn, position:position:1j]

    slice = gdd(array([xxs,yys]).T*fac, fluxes,
                    (grid_x*fac, grid_y*fac), method=inter, fill_value=0)


    return grid_x, grid_y, slice
