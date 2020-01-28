from numpy import *
from joelib.constants.constants import cc, mp, me, qe, sigT, sTd
from tqdm import *
from afterglow_dynamics import obsTime_offAxis_General_NEXP, obsTime_offAxis_General_EXP
from afterglow_properties import *
from scipy.interpolate import interp1d


# Utilities and functions for calculating GRB observables, such as light curves for different components of the emission and synthetic images ##########


def obsangle(thetas, phis, theta_obs):
        """
        Return the cosine of the observer angle for the different shockwave segments and and
        and observer at and angle theta_obs with respect to the jet axis
        (contained in yz plane)
        """
        #u_obs_x, u_obs_y, u_obs_z = 0., sin(theta_obs), cos(theta_obs)
        u_obs_y, u_obs_z = sin(theta_obs), cos(theta_obs)

        #seg_x =
        seg_y = sin(thetas)*sin(phis)
        seg_z = cos(thetas)

        #return  arccos(u_obs_x*seg_x + u_obs_y*seg_y + u_obs_z*seg_z)
        return  u_obs_y*seg_y + u_obs_z*seg_z

def obsangle_cj(thetas, phis, theta_obs):
        """
        Return the cosine of the observer angle for the different shockwave
        segments in the counter jet and observer at an angle theta_obs with respect to the jet axis
        (contained in yz plane)
        """
        #u_obs_x, u_obs_y, u_obs_z = 0., sin(theta_obs), cos(theta_obs)
        u_obs_y, u_obs_z = sin(theta_obs), cos(theta_obs)

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



def light_curve_peer_TH(jet, pp, theta_obs, obsFreqs, DD, tt0, ttf, num, Rb):

        # Takes top hat jet as input parameter!

        if type(obsFreqs)==float:
            obsFreqs  = array([obsFreqs])

        #calpha = obsangle(jet.cthetas, jet.cphis, theta_obs)
        #alpha  = arccos(calpha)

        #calpha_cj = obsangle_cj(jet.cthetas, jet.cphis, theta_obs)
        #alpha_cj  = arccos(calpha_cj)




#        if (ttf>max_Tobs or ttf>max_Tobs_cj):
#            print "ttf larger than maximum observable time. Adjusting value. "
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
            #calpha, calpha_cj = obsangle(theta_cell, phi_cell, theta_obs), obsangle_cj(theta_cell, phi_cell, theta_obs)
            #alpha,  alpha_cj = arccos(calpha), arccos(calpha_cj)

            onAxisTint = interp1d(jet.RRs, jet.TTs)

            if jet.aa >= 0:
                # For laterally expanding shells
                theta_cellR = ones([jet.steps])*jet.cthetas[0,layer-1] + 0.5*arcsin(sin(jet.thetas[:, layer-1])-sin(jet.initJoAngle))
                calphaR, calphaR_cj = obsangle(theta_cellR, phi_cell, theta_obs), obsangle_cj(theta_cellR, phi_cell, theta_obs)
                alphaR, alphaR_cj   = arccos(calphaR), arccos(calphaR_cj)
                alphaRI, alphaR_cjI = interp1d(jet.RRs, alphaR), interp1d(jet.RRs, alphaR_cj)
                ttobs = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs, calphaR)
                ttobs_cj = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs, calphaR_cj)

            else:
                # For collimated shells
                calpha, calpha_cj = obsangle(jet.cthetas[0,layer-1], phi_cell, theta_obs), obsangle(jet.cthetas[0,layer-1], phi_cell, theta_obs)
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
            GamObs = jet.GamInt(Robs)
            BetaObs = sqrt(1.-GamObs**(-2.))
            #if jet.evolution == 'adiabatic':
            #    onAxisTobs = obsTime_onAxis_adiabatic(Robs, BetaObs)
            #elif jet.evolution == 'peer':
            #    onAxisTobs = obsTime_onAxis_integrated(Robs, GamObs, BetaObs)
            onAxisTobs = onAxisTint(Robs)
            #angExt = jet.angExtI(Robs)
            nE = jet.neI(Robs)

            Rint_cj = interp1d(ttobs_cj, jet.RRs)
            Robs_cj= Rint(tts[filTM_cj][filTm_cj])
            GamObs_cj = jet.GamInt(Robs_cj)
            BetaObs_cj = sqrt(1.-GamObs_cj**(-2.))
            onAxisTobs_cj = onAxisTint(Robs_cj)
            #angExt_cj = jet.angExtI(Robs_cj)
            nE_cj = jet.neI(Robs_cj)

            if jet.aa>=0:
                alpha, alpha_cj  = alphaRI(Robs), alphaR_cjI(Robs)
                calpha, calpha_cj = cos(alpha), cos(alpha_cj)
            else:
                alpha, alpha_cj = ones(len(Robs))*alpha, ones(len(Robs))*alpha_cj
                calpha, calpha_cj = ones(len(Robs))*calpha, ones(len(Robs))*calpha_cj



            # Forward shock stuff, principal jet
            Bfield = Bfield_modified(GamObs, BetaObs, jet.nn, jet.epB)
            gamMobs, nuMobs = minGam_modified(GamObs, jet.epE, jet.epB, jet.nn, pp, Bfield, jet.Xp)
            gamCobs, nuCobs = critGam_modified(GamObs, jet.epE, jet.epB, jet.nn, pp, Bfield, onAxisTobs)
            Fnuobs = fluxMax_modified(Robs, GamObs, nE, Bfield, jet.PhiP)

            # Forward shock, counter-jet stuff
            Bfield_cj = Bfield_modified(GamObs_cj, BetaObs_cj, jet.nn, jet.epB)
            gamMobs_cj, nuMobs_cj = minGam_modified(GamObs_cj, jet.epE, jet.epB, jet.nn, pp, Bfield_cj, jet.Xp)
            gamCobs_cj, nuCobs_cj = critGam_modified(GamObs_cj, jet.epE, jet.epB, jet.nn, pp, Bfield_cj, onAxisTobs_cj)
            Fnuobs_cj = fluxMax_modified(Robs_cj, GamObs_cj, nE_cj, Bfield_cj, jet.PhiP)


            # Reverse shock stuff
            nuM_RS, nuC_RS, Fnu_RS = params_tt_RS(jet, onAxisTobs, Rb)


            dopFacs =  dopplerFactor(calpha, BetaObs)
            #afac = angExt/maximum(angExt*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))

            dopFacs_cj =  dopplerFactor(calpha_cj, BetaObs_cj)
            #afac_cj = angExt_cj/maximum(angExt_cj*ones(num)[filTM_cj][filTm_cj], 2.*pi*(1.-cos(1./GamObs_cj)))

            for freq in obsFreqs:
                fil1, fil2 = where(gamMobs<=gamCobs)[0], where(gamMobs>gamCobs)[0]
                fil3, fil4 = where(nuM_RS<=nuC_RS)[0], where(nuM_RS>nuC_RS)[0]
                fil5, fil6 = where(nuMobs_cj<=nuCobs_cj)[0], where(nuMobs_cj>nuCobs_cj)[0]


                freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                freqs_cj = freq/dopFacs_cj

                light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                        (GamObs[fil1]*(1.-BetaObs[fil1]*calpha[fil1]))**(-3.)  * FluxNuSC_arr(jet.pp, nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))#*calpha
                #light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                #                                    afac[fil2] * dopFacs[fil2]**3. * FluxNuFC_arr(jet, nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))*calpha

                light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] + (
                                        (GamObs[fil3]*(1.-BetaObs[fil3]*calpha[fil3]))**(-3.) * FluxNuSC_arr(jet.pp, nuM_RS[fil3], nuC_RS[fil3], Fnu_RS[fil3], freqs[fil3]))#*calpha
                #light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] + (
                #                                    afac[fil4] * dopFacs[fil4]**3. * FluxNuFC_arr(jet, nuM_RS[fil4], nuC_RS[fil4], Fnu_RS[fil4], freqs[fil4]))*calpha
                light_curve_CJ[obsFreqs==freq, filTM_cj[filTm_cj][fil5]] = light_curve_CJ[obsFreqs==freq, filTM_cj[filTm_cj][fil5]] + (
                                        (GamObs_cj[fil5]*(1.-BetaObs_cj[fil5]*calpha_cj[fil5]))**(-3.) * FluxNuSC_arr(jet.pp, nuMobs_cj[fil5], nuCobs_cj[fil5], Fnuobs_cj[fil5], freqs_cj[fil5]))#*calpha



        #return tts, 2.*light_curve, 2.*light_curve_RS
        return tts, light_curve/(DD**2.), light_curve_RS/(DD**2.), light_curve_CJ/(DD**2.)




def light_curve_peer_SJ(jet, pp, theta_obs, obsFreqs, DD, tt0, ttf, num, Rb):

        # Takes top hat jet as input parameter!

        if type(obsFreqs)==float:
            obsFreqs  = array([obsFreqs])

        #calpha = obsangle(jet.cthetas, jet.cphis, theta_obs)
        #alpha  = arccos(calpha)

        #calpha_cj = obsangle_cj(jet.cthetas, jet.cphis, theta_obs)
        #alpha_cj  = arccos(calpha_cj)




#        if (ttf>max_Tobs or ttf>max_Tobs_cj):
#            print "ttf larger than maximum observable time. Adjusting value. "
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
            #calpha, calpha_cj = obsangle(theta_cell, phi_cell, theta_obs), obsangle_cj(theta_cell, phi_cell, theta_obs)
            #alpha,  alpha_cj = arccos(calpha), arccos(calpha_cj)


            GamInt = jet.GamInt[layer-1]
            onAxisTint  = jet.TTInt[layer-1]
            if jet.aa >= 0:
                # For laterally expanding shells
                theta_cellR = ones([jet.steps])*jet.cthetas[0,layer-1] + 0.5*arcsin(sin(jet.thetas[:, layer-1])-sin(jet.initJoAngle))
                calphaR, calphaR_cj = obsangle(theta_cellR, phi_cell, theta_obs), obsangle_cj(theta_cellR, phi_cell, theta_obs)
                alphaR, alphaR_cj   = arccos(calphaR), arccos(calphaR_cj)
                alphaRI, alphaR_cjI = interp1d(jet.RRs, alphaR), interp1d(jet.RRs, alphaR_cj)
                ttobs = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs[:,layer-1], calphaR)
                ttobs_cj = obsTime_offAxis_General_EXP(jet.RRs, jet.TTs[:,layer-1], calphaR_cj)

            else:
                # For collimated shells
                calpha, calpha_cj = obsangle(jet.cthetas0[layer-1], phi_cell, theta_obs), obsangle(jet.cthetas0[layer-1], phi_cell, theta_obs)
                alpha,  alpha_cj    = arccos(calpha), arccos(calpha_cj)
                ttobs = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs[:,layer-1], alpha)
                ttobs_cj = obsTime_offAxis_General_NEXP(jet.RRs, jet.TTs[:,layer-1], alpha_cj)



            #ttobs = obsTime_offAxis_UR(jet.RRs, jet.TTs, jet.Betas, alpha)

            filTM  = where(tts<=max(ttobs))[0]
            filTm  = where(tts[filTM]>=min(ttobs))[0]
            filTM_cj  = where(tts<=max(ttobs_cj))[0]
            filTm_cj  = where(tts[filTM_cj]>=min(ttobs_cj))[0]
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
            onAxisTobs = onAxisTint(Robs)
            #angExt = jet.angExtI(Robs)
            nE = jet.neI[layer-1](Robs)

            Rint_cj = interp1d(ttobs_cj, jet.RRs)
            Robs_cj= Rint(tts[filTM_cj][filTm_cj])
            GamObs_cj = jet.GamInt[layer-1](Robs_cj)
            BetaObs_cj = sqrt(1.-GamObs_cj**(-2.))
            onAxisTobs_cj = onAxisTint(Robs_cj)
            #angExt_cj = jet.angExtI(Robs_cj)
            nE_cj = jet.neI[layer-1](Robs_cj)

            if jet.aa>=0:
                alpha, alpha_cj  = alphaRI(Robs), alphaR_cjI(Robs)
                calpha, calpha_cj = cos(alpha), cos(alpha_cj)
            else:
                alpha, alpha_cj = ones(len(Robs))*alpha, ones(len(Robs))*alpha_cj
                calpha, calpha_cj = ones(len(Robs))*calpha, ones(len(Robs))*calpha_cj



            # Forward shock stuff, principal jet
            Bfield = Bfield_modified(GamObs, BetaObs, jet.nn, jet.epB)
            gamMobs, nuMobs = minGam_modified(GamObs, jet.epE, jet.epB, jet.nn, pp, Bfield, jet.Xp)
            gamCobs, nuCobs = critGam_modified(GamObs, jet.epE, jet.epB, jet.nn, pp, Bfield, onAxisTobs)
            Fnuobs = fluxMax_modified(Robs, GamObs, nE, Bfield, jet.PhiP)

            # Forward shock, counter-jet stuff
            Bfield_cj = Bfield_modified(GamObs_cj, BetaObs_cj, jet.nn, jet.epB)
            gamMobs_cj, nuMobs_cj = minGam_modified(GamObs_cj, jet.epE, jet.epB, jet.nn, pp, Bfield_cj, jet.Xp)
            gamCobs_cj, nuCobs_cj = critGam_modified(GamObs_cj, jet.epE, jet.epB, jet.nn, pp, Bfield_cj, onAxisTobs_cj)
            Fnuobs_cj = fluxMax_modified(Robs_cj, GamObs_cj, nE_cj, Bfield_cj, jet.PhiP)


            # Reverse shock stuff
            nuM_RS, nuC_RS, Fnu_RS = params_tt_RS_SJ(jet, onAxisTobs, layer-1, Rb)


            dopFacs =  dopplerFactor(calpha, BetaObs)
            #afac = angExt/maximum(angExt*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))

            dopFacs_cj =  dopplerFactor(calpha_cj, BetaObs_cj)
            #afac_cj = angExt_cj/maximum(angExt_cj*ones(num)[filTM_cj][filTm_cj], 2.*pi*(1.-cos(1./GamObs_cj)))

            for freq in obsFreqs:
                fil1, fil2 = where(gamMobs<=gamCobs)[0], where(gamMobs>gamCobs)[0]
                fil3, fil4 = where(nuM_RS<=nuC_RS)[0], where(nuM_RS>nuC_RS)[0]
                fil5, fil6 = where(nuMobs_cj<=nuCobs_cj)[0], where(nuMobs_cj>nuCobs_cj)[0]


                freqs = freq*dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                freqs_cj = freq*dopFacs_cj




                light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                        (GamObs[fil1]*(1.-BetaObs[fil1]*calpha[fil1]))**(-3.)  * FluxNuSC_arr(jet.pp, nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))#*calpha
                #light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                #                                    afac[fil2] * dopFacs[fil2]**3. * FluxNuFC_arr(jet, nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))*calpha

                light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] + (
                                        (GamObs[fil3]*(1.-BetaObs[fil3]*calpha[fil3]))**(-3.) * FluxNuSC_arr(jet.pp, nuM_RS[fil3], nuC_RS[fil3], Fnu_RS[fil3], freqs[fil3]))#*calpha
                #light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] + (
                #                                    afac[fil4] * dopFacs[fil4]**3. * FluxNuFC_arr(jet, nuM_RS[fil4], nuC_RS[fil4], Fnu_RS[fil4], freqs[fil4]))*calpha
                light_curve_CJ[obsFreqs==freq, filTM_cj[filTm_cj][fil5]] = light_curve_CJ[obsFreqs==freq, filTM_cj[filTm_cj][fil5]] + (
                                        (GamObs_cj[fil5]*(1.-BetaObs_cj[fil5]*calpha_cj[fil5]))**(-3.) * FluxNuSC_arr(jet.pp, nuMobs_cj[fil5], nuCobs_cj[fil5], Fnuobs_cj[fil5], freqs_cj[fil5]))#*calpha



        #return tts, 2.*light_curve, 2.*light_curve_RS
        return tts, light_curve/(DD**2.), light_curve_RS/(DD**2.), light_curve_CJ/(DD**2.)
