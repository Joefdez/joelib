from numpy import *
import joelib.constants.constants as cts
from synchrotron_afterglow import *
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from tqdm import tqdm



class jetHeadUD(adiabatic_afterglow):


###############################################################################################
# Methods for initializing the cells in the jet head
###############################################################################################

    def __init__(self, EE,  Gam0, nn, epE, epB, pp, DD, steps, evolution, nlayers, joAngle, shell_type='thin', Rb=1.):#, obsAngle=0.0):

        self.nlayers  = nlayers                         # Number of layers for the partition
        #self.nn1      = nn1                            # Number of cells in the first layer
        self.__totalCells()                             # obtain self.ncells
        self.joAngle  = joAngle                         # Jet opening angle
        #self.obsAngle = obsAngle                        # Angle of jet axis with respect to line of sight
        self.angExt   = 2.*pi*(1.-cos(joAngle))      # Solid area covered by the jet head
        self.cellSize = self.angExt/self.ncells            # Angular size of each cell
        self.__makeCells()                              # Generate the cells: calculate the angular positions of the shells
        adiabatic_afterglow.__init__(self, EE, Gam0, nn, epE, epB, pp, DD, steps, evolution, shell_type, Rb)
        self.ee       = EE/self.ncells                  # Energy per cell

    def __makeCells(self):

        """
        This method generates the individual cells: positions of borders between cells
        and angular positions of the cells themselves.
        """

        self.layer = array([])
        self.thetas = array([])
        self.phis   = array([])
        self.cthetas = array([])
        self.cphis   = array([])

        fac1 = arange(0,self.nlayers+1)/float(self.nlayers)        # Numerical factor for use during execution
        self.thetas = 2.*arcsin(fac1*sin(self.joAngle/4.))         # Calculate the propagation angle with respect to jet axis

        for ii in range(self.nlayers):                             # Loop over layers and populate the arrays
            num = self.cellsInLayer(ii)
            self.phis    = append(self.phis, arange(0,num+1)*2.*pi/num)     # Phi value of the edges
            self.layer   = append(self.layer,ones(num)*(ii+1))            # Layer on which the cells are
            self.cthetas = append(self.cthetas,ones(num)*0.5*(self.thetas[ii]+self.thetas[ii+1]))   # Central theta values of the cells
            self.cphis   = append(self.cphis,(arange(0,num)+0.5)*2.*pi/num )    # Central phi values of the cells

            #num = int(round(self.cellsInLayer(ii)/2))
            #self.layer   = append(self.layer,ones(num+1)*(ii+1))            # Layer on which the phi edges are
            #self.phis    = append(self.phis, arange(0,num+1)*2.*pi/num)     # Phi value of the edges
            #self.cthetas = append(self.cthetas,ones(num)*0.5*(self.thetas[ii]+self.thetas[ii+1]))   # Central theta values
            #self.cphis   = append(self.cphis,(arange(0,num)+0.5)*pi/num )    # Central phi values


    def __totalCells(self):
        tot = 0
        for ii in range(0,self.nlayers):
            tot = tot + self.cellsInLayer(ii)
            #tot = tot + int(round(self.cellsInLayer(ii)/2))
        self.ncells = tot



###############################################################################################
# Methods used by initializers and for getting different physics and general methods not used by initializers
###############################################################################################

    def cellsInLayer(self, ii):
        """
        Return number of cells in layer ii
        """
        return (2*ii+1)


    def obsangle(self, theta_obs):
        """
        Return the cosine of the observer angle for the different shockwave segments and and
        and observer at and angle theta_obs with respect to the jet axis
        (contained in yz plane)
        """
        #u_obs_x, u_obs_y, u_obs_z = 0., sin(theta_obs), cos(theta_obs)
        u_obs_y, u_obs_z = sin(theta_obs), cos(theta_obs)

        #seg_x =
        seg_y = sin(self.cthetas)*sin(self.cphis)
        seg_z = cos(self.cthetas)

        #return  arccos(u_obs_x*seg_x + u_obs_y*seg_y + u_obs_z*seg_z)
        return  u_obs_y*seg_y + u_obs_z*seg_z

    def dopplerFactor(self, cosa, beta):
        """
        Calculate the doppler factors of the different jethead segments
        cosa -> cosine of observeration angle, obtained using obsangle
        """


        return (1.-beta)/(1.-beta*cosa)



    def light_curve_adiabatic(self, theta_obs, obsFreqs, tt0, ttf, num, Rb):


        if type(obsFreqs)==float:
            obsFreqs  = array([obsFreqs])

        calpha = self.obsangle(theta_obs)
        alpha  = arccos(calpha)

        if self.evolution == 'adiabatic':
            max_Tobs = max(obsTime_offAxis_UR(self.RRs, self.TTs, self.Betas, max(alpha)))/cts.sTd
        elif self.evolution == 'peer':
            max_Tobs = max(obsTime_offAxis_General(self.RRs, self.TTs, max(alpha)))/cts.sTd
            #max_Tobs = max(obsTime_offAxis_UR(self.RRs, self.TTs, self.Betas, max(alpha)))/cts.sTd


        if ttf>max_Tobs:
            print "ttf larger than maximum observable time. Adjusting value. "
            ttf = max_Tobs

        lt0 = log10(tt0*cts.sTd) # Convert to seconds and then logspace
        ltf = log10(ttf*cts.sTd) # Convert to seconds and then logspace

        tts = logspace(lt0, ltf+(ltf-lt0)/num, num) # Timeline on which the flux is evaluated.


        light_curve      = zeros([len(obsFreqs), num])
        light_curve_RS   = zeros([len(obsFreqs), num])


        for ii in tqdm(range(self.ncells)):
        #for ii in range(self.ncells):
            ttobs = obsTime_offAxis_UR(self.RRs, self.TTs, self.Betas, alpha[ii])


            filTM  = where(tts<=max(ttobs))[0]
            filTm  = where(tts[filTM]>=min(ttobs))[0]
            #print(len(tts[filT]))

            Rint = interp1d(ttobs, self.RRs)
            Robs = Rint(tts[filTM][filTm])
            GamObs = self.GamInt(Robs)
            BetaObs = sqrt(1.-GamObs**(-2.))
            onAxisTint = interp1d(self.RRs, self.TTs)
            #if self.evolution == 'adiabatic':
            #    onAxisTobs = obsTime_onAxis_adiabatic(Robs, BetaObs)
            #elif self.evolution == 'peer':
            #    onAxisTobs = obsTime_onAxis_integrated(Robs, GamObs, BetaObs)
            onAxisTobs = onAxisTint(Robs)

            # Forward shock stuff
            #gamMobs, gamCobs = self.gamMI(Robs), self.gamCI(Robs)
            #nuMobs, nuCobs   = self.nuMI(Robs), self.nuCI(Robs)
            #Fnuobs = self.FnuMI(Robs)
            Bfield = sqrt(32.*pi*self.nn*self.epB*cts.mp)*cts.cc*GamObs
            gamMobs, nuMobs = minGam(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield)
            gamCobs, nuCobs = critGam(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, onAxisTobs)
            Fnuobs = fluxMax(Robs, GamObs, self.nn, Bfield, self.DD)




            # Reverse shock stuff
            nuM_RS, nuC_RS, Fnu_RS = params_tt_RS(self, onAxisTobs, Rb)


            dopFacs =  self.dopplerFactor(calpha[ii], sqrt(1.-GamObs**(-2)))
            afac = self.cellSize/maximum(self.cellSize*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))


            for freq in obsFreqs:
                fil1, fil2 = where(gamMobs<=gamCobs)[0], where(gamMobs>gamCobs)[0]
                fil3, fil4 = where(nuM_RS<=nuC_RS)[0], where(nuM_RS>nuC_RS)[0]
                freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                #print shape(freqs), shape(freqs[fil1]), shape(nuMobs[fil1]), shape(nuCobs[fil1]), shape(Fnuobs[fil1]), shape(afac[fil1]), shape(calpha)
                #print shape(light_curve[obsFreqs==freq, filT]), shape([fil1])
                #print fil1
                light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                                    afac[fil1] * dopFacs[fil1]**3. * FluxNuSC_arr(self, nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))*calpha[ii]
                #light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                #                                   afac[fil2] * dopFacs[fil2]**3. * FluxNuFC_arr(self, nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))*calpha[ii]

                light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] + (
                                                    afac[fil3] * dopFacs[fil3]**3. * FluxNuSC_arr(self, nuM_RS[fil3], nuC_RS[fil3], Fnu_RS[fil3], freqs[fil3]))*calpha[ii]
                #light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] + (
                #                                    afac[fil4] * dopFacs[fil4]**3. * FluxNuFC_arr(self, nuM_RS[fil4], nuC_RS[fil4], Fnu_RS[fil4], freqs[fil4]))*calpha[ii]
                #cont1 = afac[fil1] * dopFacs[fil1]**3. * self.FluxNuSC_arr(nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1])*calpha[ii]
                #cont2 = afac[fil2] * dopFacs[fil2]**3. * self.FluxNuFC_arr(nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2])*calpha[ii]

                #light_curve[obsFreqs==freq, filT][fil1] += cont1
                #light_curve[obsFreqs==freq, filT][fil2] += cont2


        #return tts, 2.*light_curve, 2.*light_curve_RS
        return tts, light_curve, light_curve_RS

    def light_curve_peer(self, theta_obs, obsFreqs, tt0, ttf, num, Rb):


        if type(obsFreqs)==float:
            obsFreqs  = array([obsFreqs])

        calpha = self.obsangle(theta_obs)
        alpha  = arccos(calpha)


        max_Tobs = max(obsTime_offAxis_General(self.RRs, self.TTs, max(alpha)))/cts.sTd
        #max_Tobs = max(obsTime_offAxis_UR(self.RRs, self.TTs, self.Betas, max(alpha)))/cts.sTd


        if ttf>max_Tobs:
            print "ttf larger than maximum observable time. Adjusting value. "
            ttf = max_Tobs

        lt0 = log10(tt0*cts.sTd) # Convert to seconds and then logspace
        ltf = log10(ttf*cts.sTd) # Convert to seconds and then logspace

        tts = logspace(lt0, ltf+(ltf-lt0)/num, num) # Timeline on which the flux is evaluated.


        light_curve      = zeros([len(obsFreqs), num])
        light_curve_RS   = zeros([len(obsFreqs), num])


        for ii in tqdm(range(self.ncells)):
        #for ii in range(self.ncells):

            ttobs = obsTime_offAxis_General(self.RRs, self.TTs, alpha[ii])
            #ttobs = obsTime_offAxis_UR(self.RRs, self.TTs, self.Betas, alpha[ii])

            filTM  = where(tts<=max(ttobs))[0]
            filTm  = where(tts[filTM]>=min(ttobs))[0]
            #print(len(tts[filT]))

            Rint = interp1d(ttobs, self.RRs)
            Robs = Rint(tts[filTM][filTm])
            GamObs = self.GamInt(Robs)
            BetaObs = sqrt(1.-GamObs**(-2.))
            onAxisTint = interp1d(self.RRs, self.TTs)
            #if self.evolution == 'adiabatic':
            #    onAxisTobs = obsTime_onAxis_adiabatic(Robs, BetaObs)
            #elif self.evolution == 'peer':
            #    onAxisTobs = obsTime_onAxis_integrated(Robs, GamObs, BetaObs)
            onAxisTobs = onAxisTint(Robs)

            # Forward shock stuff
            #gamMobs, gamCobs = self.gamMI(Robs), self.gamCI(Robs)
            #nuMobs, nuCobs   = self.nuMI(Robs), self.nuCI(Robs)
            #Fnuobs = self.FnuMI(Robs)
            #Bfield = sqrt(32.*pi*cts.mp*self.nn*self.epB*GamObs*(GamObs-1.))*cts.cc
            Bfield = Bfield_modified(GamObs, BetaObs, self.nn, self.epB)
            gamMobs, nuMobs = minGam_modified(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, self.Xp)
            gamCobs, nuCobs = critGam_modified(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, onAxisTobs)
            Fnuobs = fluxMax_modified(Robs, GamObs, self.nn, Bfield, self.DD, self.PhiP)


            # Reverse shock stuff
            nuM_RS, nuC_RS, Fnu_RS = params_tt_RS(self, onAxisTobs, Rb)


            dopFacs =  self.dopplerFactor(calpha[ii], sqrt(1.-GamObs**(-2)))
            afac = self.cellSize/maximum(self.cellSize*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))


            for freq in obsFreqs:
                fil1, fil2 = where(gamMobs<=gamCobs)[0], where(gamMobs>gamCobs)[0]
                fil3, fil4 = where(nuM_RS<=nuC_RS)[0], where(nuM_RS>nuC_RS)[0]
                freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                #print shape(freqs), shape(freqs[fil1]), shape(nuMobs[fil1]), shape(nuCobs[fil1]), shape(Fnuobs[fil1]), shape(afac[fil1]), shape(calpha)
                #print shape(light_curve[obsFreqs==freq, filT]), shape([fil1])
                #print fil1
                light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                        self.cellSize * (GamObs[fil1]*(1.-BetaObs[fil1]*calpha[ii]))**(-3.)  * FluxNuSC_arr(self, nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))#*calpha[ii]
                #light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                #                                    afac[fil2] * dopFacs[fil2]**3. * FluxNuFC_arr(self, nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))*calpha[ii]

                light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] + (
                                        self.cellSize * (GamObs[fil3]*(1.-BetaObs[fil3]*calpha[ii]))**(-3.) * FluxNuSC_arr(self, nuM_RS[fil3], nuC_RS[fil3], Fnu_RS[fil3], freqs[fil3]))#*calpha[ii]
                #light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] + (
                #                                    afac[fil4] * dopFacs[fil4]**3. * FluxNuFC_arr(self, nuM_RS[fil4], nuC_RS[fil4], Fnu_RS[fil4], freqs[fil4]))*calpha[ii]
                #cont1 = afac[fil1] * dopFacs[fil1]**3. * self.FluxNuSC_arr(nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1])*calpha[ii]
                #cont2 = afac[fil2] * dopFacs[fil2]**3. * self.FluxNuFC_arr(nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2])*calpha[ii]

                #light_curve[obsFreqs==freq, filT][fil1] += cont1
                #light_curve[obsFreqs==freq, filT][fil2] += cont2



        #return tts, 2.*light_curve, 2.*light_curve_RS
        return tts, light_curve, light_curve_RS


    def lightCurve_interp(self, theta_obs, obsFreqs, tt0, ttf, num, Rb):
        if self.evolution == "adiabatic":
                tts, light_curve, light_curve_RS = self.light_curve_adiabatic(theta_obs, obsFreqs, tt0, ttf, num, Rb)
        elif self.evolution == "peer":
                tts, light_curve, light_curve_RS = self.light_curve_peer(theta_obs, obsFreqs, tt0, ttf, num, Rb)

        return tts, light_curve, light_curve_RS


    def skymap(self, theta_obs, tt_obs, freq, nx, ny, xx0, yy0):

        calpha = self.obsangle(theta_obs)
        alpha  = arccos(calpha)



        TTs, RRs, Gams, Betas = zeros(self.ncells), zeros(self.ncells), zeros(self.ncells), zeros(self.ncells)
        nuMs, nuCs, fluxes    = zeros(self.ncells), zeros(self.ncells), zeros(self.ncells)
        im_xxs = sin(self.cthetas)*cos(self.cphis)
        im_yys = cos(theta_obs)*sin(self.cthetas)*sin(self.cphis) - sin(theta_obs)*cos(self.cthetas)


        if self.evolution == 'adiabatic':
            Tint = interp1d(self.RRs, self.TTs)
            for ii in tqdm(range(self.ncells)):
                ttobs = obsTime_offAxis_UR(self.RRs, self.TTs, self.Betas, alpha[ii])
                Rint = interp1d(ttobs, self.RRs)
                RRs[ii] = Rint(tt_obs)
                TTs[ii] = Tint(RRs[ii])

                Gams[ii] = self.GamInt(RRs[ii])
                Betas[ii] = sqrt(1.-Gams[ii]**(-2.))

            Bf        = (32.*pi*self.nn*self.epB*cts.mp)**(1./2.) * Gams*cts.cc
            gamM, nuM = minGam(Gams, self.epE, self.epB, self.nn, self.pp, Bf)
            gamC, nuC = critGam(Gams, self.epE, self.epB, self.nn, self.pp, Bf, TTs)
            fMax      = fluxMax(RRs, Gams, self.nn, Bf, self.DD)

            dopFacs =  self.dopplerFactor(calpha, sqrt(1.-Gams**(-2)))
            afac = self.cellSize/maximum(self.cellSize*ones(len(Gams)), 2.*pi*(1.-cos(1./Gams)))
            obsFreqs = freq/dopFacs

            fluxes = (self.DD**2./(calpha*self.cellSize*RRs**2.)) *afac * dopFacs**3. * FluxNuSC_arr(self, nuM, nuC, fMax, obsFreqs)*1./calpha
            #fluxes = afac * dopFacs**3. * FluxNuSC_arr(self, nuM, nuC, fMax, obsFreqs)*calpha

        elif self.evolution == 'peer':
            Tint = interp1d(self.RRs, self.TTs)
            for ii in tqdm(range(self.ncells)):
                ttobs = obsTime_offAxis_General(self.RRs, self.TTs, alpha[ii])
                Rint = interp1d(ttobs, self.RRs)
                RRs[ii] = Rint(tt_obs)
                TTs[ii] = Tint(RRs[ii])

                Gams[ii] = self.GamInt(RRs[ii])
                Betas[ii] = sqrt(1.-Gams[ii]**(-2.))

            Bf        = Bfield_modified(Gams, Betas, self.nn, self.epB)
            gamM, nuM = minGam_modified(Gams, self.epE, self.epB, self.nn, self.pp, Bf, self.Xp)
            gamC, nuC = critGam_modified(Gams, self.epE, self.epB, self.nn, self.pp, Bf, TTs)
            fMax      = fluxMax_modified(RRs, Gams, self.nn, Bf, self.DD, self.PhiP)

            dopFacs =  self.dopplerFactor(calpha, sqrt(1.-Gams**(-2)))
            obsFreqs = freq/dopFacs

            #fluxes = (self.DD/self.cellSize*RRs)**2. * self.cellSize * (Gams*(1.-Betas*calpha))**(-3.) * FluxNuSC_arr(self, nuM, nuC, fMax, obsFreqs)*1./calpha
            fluxes = (self.DD**2./(calpha*self.cellSize*RRs**2.)) * self.cellSize * (Gams*(1.-Betas*calpha))**(-3.) * FluxNuSC_arr(self, nuM, nuC, fMax, obsFreqs)
            #fluxes = self.cellSize * (Gams*(1.-Betas*calpha))**(-3.) * FluxNuSC_arr(self, nuM, nuC, fMax, obsFreqs)#*calpha


        im_xxs = RRs*im_xxs
        im_yys = RRs*im_yys

        return im_xxs, im_yys, fluxes, RRs, Gams, calpha, TTs




class jetHeadGauss(jetHeadUD):

        def __init__(self, EE,  Gam0, nn, epE, epB, pp, DD, steps, evolution, nlayers, joAngle, coAngle, shell_type='thin', Rb=1.):

            # In this case, EE refers to the total energy and Gam0 to the central Gam0 value

            self.coAngle = coAngle
            jetHeadUD.__init__(self, EE,  Gam0, nn, epE, epB, pp, DD, steps, evolution, nlayers, joAngle, shell_type, Rb)
            self.__energies_and_LF()
            if self.evolution == 'adiabatic':
                self.cell_Rds = (3./(4.*pi) * 1./(cts.cc**2.*cts.mp) *
                            self.cell_EEs/(self.nn*self.cell_Gam0s**2.))**(1./3.)
                self.cell_Tds = self.cell_Rds/(cts.cc*self.cell_Beta0s) * (1.-self.cell_Beta0s)
                #self.cell_Tds  = self.cell_Rds/(2.*cts.cc*self.cell_Gam0s**2.)
                #self.Rd/(2.*self.Gam0**2 * cts.cc)
            elif self.evolution == 'peer':
                self.cell_Rds = (3./(4.*pi) * 1./(cts.cc**2.*cts.mp) *
                                    self.cell_EEs/(self.nn*self.cell_Gam0s**2.))**(1./3.)
                self.cell_Tds = self.cell_Rds/(cts.cc*self.cell_Beta0s) * (1.-self.cell_Beta0s)
            print "Calculating dynamical evolution"
            self.__evolve()
            print "Calculating reverse shock parmeters"
            self.__peakParamsRS_struc()


        def __energies_and_LF(self):

            #AngFacs = exp(-1.*self.cthetas**2./(2.*self.coAngle**2.))
            self.cell_EEs = self.EE * exp(-1.*self.cthetas**2./(self.coAngle**2.))    # Just for texting
            #self.cell_EEs = self.EE * exp(-1.*self.cthetas**2./(self.coAngle**2.))
            self.cell_Gam0s = (self.Gam0-1)*exp(-1.*self.cthetas**2./(2.*self.coAngle**2.))
            self.cell_Beta0s = sqrt(1.-(self.cell_Gam0s)**(-2.))

        def __evolve(self):

            if self.evolution == 'peer':
                self.RRs, self.Gams, self.Betas = self.evolve_relad_struct()
                self.TTs = self.obsTime_onAxis_struct()
                self.Bfield = Bfield_modified(self.Gams, self.Betas, self.nn, self.epB)

            elif self.evolution == 'adiabatic':
                self.RRs, self.Gams, self.Betas = self.evolve_ad_struct()
                self.TTs = self.obsTime_onAxis_struct()
                self.Bfield = (32.*pi*cts.mp*self.epB*self.nn)**(1./2.)*self.Gams*cts.cc

        def __peakParamsRS_struc(self):

            RSpeak_nuM_struc = zeros(self.ncells)
            RSpeak_nuC_struc = zeros(self.ncells)
            RSpeak_Fnu_struc = zeros(self.ncells)


            if self.shell_type=='thin':
                print("Setting up thin shell")
                for ii in tqdm(range(self.ncells)):
                    #self.RSpeak_nuM = 9.6e14 * epE**2. * epB**(1./2.) * nn**(1./2) * Gam0**2.
                    #self.RSpeak_nuC = 4.0e16 * epB**(-3./2.) * EE**(-2./3.) * nn**(-5./6.) * Gam0**(4./3.)
                    #self.RSpeak_Fnu = 5.2 * DD**(-2.) * epB**(1./2.) * EE * nn**(1./2.) * Gam0

                    Rd, Td  = self.cell_Rds[ii], self.cell_Tds[ii]
                    #print Rd

                    if self.evolution == 'peer':
                        #print shape(self.RRs), shape(self.Gams)
                        GamsInt = interp1d(self.RRs[:], self.Gams[:,ii])
                        Gam0 = GamsInt(Rd)
                        Beta0 = sqrt(1.-Gam0**(-2.))
                        Bf = Bfield_modified(Gam0, Beta0, self.nn, self.epB)
                        gamM, nuM = minGam_modified(Gam0, self.epE, self.epB, self.nn, self.pp, Bf, self.Xp)
                        gamC, nuC = critGam_modified(Gam0, self.epE, self.epB, self.nn, self.pp, Bf, Td)
                        Fnu = fluxMax_modified(Rd, Gam0, self.nn, Bf, self.DD, self.PhiP)

                    elif self.evolution == 'adiabatic':
                        GamsInt = interp1d(self.RRs[:,ii], self.Gams[:,ii])
                        Gam0 = GamsInt(Rd)
                        Bf = (32.*pi*cts.mp*self.epB*self.nn)**(1./2.)*Gam0*cts.cc
                        gamM, nuM = minGam(Gam0, self.epE, self.epB, self.nn, self.pp, Bf)
                        gamC, nuC = critGam(Gam0, self.epE, self.epB, self.nn, self.pp, Bf, Td)
                        Fnu = fluxMax(Rd, Gam0, self.nn, Bf, self.DD)

                        #print Rd, max(self.RRs[:,ii]), min(self.RRs[:,ii]), self.cell_Gam0s[ii], self.cthetas[ii]


                    #gamM = self.epE*(self.pp-2.)/(self.pp-1.) * cts.mp/cts.me * Gam0
                    #gamC = 3.*cts.me/(16.*self.epB*cts.sigT*cts.mp*cts.cc*Gam0**3.*Td*self.nn)
                    #nuM = Gam0*gamM**2.*cts.qe*(32.*pi*cts.mp*self.epB*self.nn)**(1./2.)*Gam0*cts.cc/(2.*pi*cts.me*cts.cc)
                    #nuC = Gam0*gamC**2.*cts.qe*(32.*pi*cts.mp*self.epB*self.nn)**(1./2.)*Gam0*cts.cc/(2.*pi*cts.me*cts.cc)
                    #Fnu = self.nn**(3./2.)*Rd**3. * cts.sigT * cts.cc**3. *cts.me* (32.*pi*cts.mp*self.epB
                    #        )**(1./2.)*Gam0**2./(9.*cts.qe*self.DD**2.)


                    #RSpeak_nuM_struc[ii] = nuM/(self.cell_Gam0s[ii]**2.)
                    #RSpeak_nuC_struc[ii] = nuC
                    #RSpeak_Fnu_struc[ii] = self.cell_Gam0s[ii] * Fnu
                    RSpeak_nuM_struc[ii]  = nuM/(Gam0**2)
                    RSpeak_nuC_struc[ii]  = nuC
                    RSpeak_Fnu_struc[ii] =  Gam0*Fnu


            self.RSpeak_nuM_struc = RSpeak_nuM_struc #self.Rb**(1./2.)*RSpeak_nuM_struc
            self.RSpeak_nuC_struc = RSpeak_nuC_struc #self.Rb**(-3./2.)*RSpeak_nuC_struc
            self.RSpeak_Fnu_struc = RSpeak_Fnu_struc #self.Rb**(1./2.)*RSpeak_Fnu_struc



        def evolve_relad_struct(self):
            """
            Evolution following Pe'er 2012. Adbaiatic expansion into a cold, uniform ISM using conservation of energy in relativstic form. This solution
            transitions smoothly from the ultra-relativistic to the Newtonian regime. Modified for stuctured jet
            """
            Gam0  = self.Gam0
            Rl    = self.Rd * Gam0**(2./3.)
            RRs   = logspace(log10(self.Rd/100.), log10(Rl)+3., self.steps+1) #10
            #MMs    = 4.*pi * cts.mp*self.nn*RRs**3./3.#4./3. *pi*cts.mp*self.nn*RRs**3.
            MMs = 4./3. * pi*RRs**3. * self.nn * cts.mp
            #Gams[0,:] = self.cell_Gam0s

            #print("Calculating Gamma as a function of R for each cell")
            print("Calculating dynamical evolution for each layer")
            #for ii in tqdm(range(1,len(self.Betas))):
            #    Gams[ii,:] = rk4(dgdm_struc, self, log10(MMs[ii-1]), Gams[ii-1,:], (log10(MMs[ii])-log10(MMs[ii-1])))

            for ii in tqdm(range(self.nlayers)):
                # Set up initial conditions for the layer
                #GamEv[0] = Gams[0,self.layer==ii+1][0]
                MM0 = self.cell_EEs[self.layer==ii+1][0]/(self.cell_Gam0s[self.layer==ii+1][0]*cts.cc**2.)
                self.cell_Gam0s[self.layer==ii+1][0]
                Gams = zeros(len(RRs))

                GamEv[0] = self.cell_Gam0s[self.layer==ii+1][0]
                # Calculate dynamical evolution of the layer

                for jj in range(1, len(GamEv)):
                    GamEv[jj] = rk4(dgdm_mod, MM0, log10(MMs[jj-1]), GamEv[jj-1], (log10(MMs[jj])-log10(MMs[jj-1])))


                # Share the values with the rest of the cells of the layer
                for jj in range(self.cellsInLayer(ii)):
                    if ii==0:
                        Gams = copy(GamEv)
                    else:
                        Gams = column_stack((Gams, GamEv))

            Betas = sqrt(1.-1./Gams**2.)
            #Betas[-1] = 0.0

            return RRs, Gams, Betas

        def evolve_ad_struct(self):
            """
            Evolution following simple energy conservation for an adiabatically expanding relativistic shell. Same scaling as
            Blanford-Mckee blastwave solution. This calculation is only valid in ultrarelativstic phase.
            """
            Gam  = self.Gam0
            GamSD = 1.021
            Rsd   = Gam**(2./3.) *self.Rd / GamSD       # Radius at Lorentz factor=1.005 -> after this point use Sedov-Taylor scaling
            Rl    = self.Rd * self.Gam0**(2./3.)
            #RRs   = logspace(log10(self.Rd/100.), log10(Rl), self.steps+1) #10

            RRs = zeros([self.steps+1, self.ncells])
            Gams  = zeros([self.steps+1, self.ncells])
            Betas  = zeros([self.steps+1, self.ncells])

            Gams[0,:] = self.cell_Gam0s
            for ii in range(self.ncells):
                RRs[:,ii]   = logspace(log10(self.cell_Rds[ii]/100.), log10(0.9999*self.cell_Rds[ii] * self.cell_Gam0s[ii]**(2./3.)), self.steps+1) # All start at same point
                Gams[RRs[:,ii]<=self.cell_Rds[ii],ii]  = self.cell_Gam0s[ii]
                Gams[RRs[:,ii]>self.cell_Rds[ii], ii]  = (self.cell_Rds[ii]/RRs[RRs[:,ii]>self.cell_Rds[ii],ii])**(3./2.) * self.cell_Gam0s[ii]
                #Gams[RRs>=Rsd] = 1./sqrt( 1.-(Rsd/RRs[RRs>=Rsd])**(6.)*(1.-1./(Gams[(RRs>jet.Rd) & (RRs<Rsd)][-1]**2.)))
                #Gams[RRs>=jet.Rd] = odeint(jet.dgdr, jet.Gam0, RRs[RRs>=jet.Rd])[:,0]
                #Gams[RRs>=jet.Rd] = odeint(jet.dgdr, jet.Gam0, RRs[RRs>=jet.Rd])[:,0]
                Betas[RRs[:,ii]<=self.cell_Rds[ii],ii] = sqrt(1.-(1./self.cell_Gam0s[ii])**2.)
                Betas[RRs[:,ii]>self.cell_Rds[ii], ii] = sqrt(1.-(1./Gams[RRs[:,ii]>self.cell_Rds[ii], ii])**2.)

            Betas[-1,:] = 0.
            #Gams[Gams<=1.] = 1.


            return RRs, Gams, Betas

        def obsTime_onAxis_struct(self):
            """
            On-axis observer times calculated for each individual cell
            """


            print("Calculating on-axis observerd time for each cell")
            #for ii in tqdm(range(1,len(self.Betas))):
            if self.evolution == "adiabatic":
                for layer in range(self.nlayers):
                    if layer==0:
                        TTs = obsTime_onAxis_adiabatic(self.RRs, self.Gams[:, layer], self.Betas[:, layer])
                    else:
                        layerTime = obsTime_onAxis_adiabatic(self.RRs, self.Gams[:, self.layer==layer+1][:,0],
                                                                        self.Betas[:, self.layer==layer+1][:,0])
                        for cell in range(self.cellsInLayer(layer)):
                            TTs = column_stack((TTs, layerTime))

            elif self.evolution == "peer":
                for layer in range(self.nlayers):
                    if layer==0:
                        TTs = obsTime_onAxis_integrated(self.RRs, self.Gams[:, layer], self.Betas[:, layer])
                    else:
                        layerTime = obsTime_onAxis_integrated(self.RRs, self.Gams[:, self.layer==layer+1][:,0],
                                                                                self.Betas[:, self.layer==layer+1][:,0])
                        for cell in range(self.cellsInLayer(layer)):
                            TTs = column_stack((TTs, layerTime))


            return TTs

        def params_tt_RS(self, tt, ii, Rb):

            if type(tt) == 'float': tt = array([tt])
            fil1, fil2 = where(tt<=self.cell_Tds[ii])[0], where(tt>self.cell_Tds[ii])[0]
            #print ii, len(tt)
            nuM = zeros(len(tt))
            nuC = zeros(len(tt))
            fluxMax = zeros(len(tt))
            #print len(nuM), len(nuC), len()

            nuM[fil1]     = self.RSpeak_nuM_struc[ii]*(tt[fil1]/self.cell_Tds[ii])**(6.)
            nuC[fil1]     = self.RSpeak_nuC_struc[ii]*(tt[fil1]/self.cell_Tds[ii])**(-2.)
            fluxMax[fil1] = self.RSpeak_Fnu_struc[ii]*(tt[fil1]/self.cell_Tds[ii])**(3./2.)   # Returns fluxes in Jy

            nuM[fil2]     = self.RSpeak_nuM_struc[ii]*(tt[fil2]/self.cell_Tds[ii])**(-54./35.)
            nuC[fil2]     = self.RSpeak_nuC_struc[ii]*(tt[fil2]/self.cell_Tds[ii])**(4./35.)
            fluxMax[fil2] = self.RSpeak_Fnu_struc[ii]*(tt[fil2]/self.cell_Tds[ii])**(-34./35.) # Returns fluxes in Jy

            return Rb**(1./2.)*nuM, Rb**(-3./2.)*nuC, Rb**(1./2.)*fluxMax


        def light_curve_adiabatic(self, theta_obs, obsFreqs, tt0, ttf, num, Rb):
            if type(obsFreqs)==float:
                obsFreqs  = array([obsFreqs])

            calpha = self.obsangle(theta_obs)
            alpha  = arccos(calpha)


            Tfil = self.TTs[:,-1]== max(self.TTs[:,-1])

            max_Tobs = self.RRs[Tfil, -1]/(self.Betas[Tfil,-1]*cts.cc) * (1.-self.Betas[Tfil,-1]*cos(max(alpha)))
            #max_Tobs_oa = max(self.TTs[:,-1])
            #max_Tobs = max(obsTime_offAxis(self, self.RRs, self.TTs[:,alpha==max(alpha)][:,0], max(alpha)))/cts.sTd
            if ttf>max_Tobs:
                print "ttf larger than maximum observable time. Adjusting value. "
                ttf = max_Tobs

            lt0 = log10(tt0*cts.sTd) # Convert to seconds and then logspace
            ltf = log10(ttf*cts.sTd) # Convert to seconds and then logspace

            tts = logspace(lt0, ltf+(ltf-lt0)/num, num) # Timeline on which the flux is evaluated.


            light_curve    = zeros([len(obsFreqs), num])
            light_curve_RS = zeros([len(obsFreqs), num])

            for ii in tqdm(range(self.ncells)):
            #for ii in range(self.ncells):
                """
                ttobs = obsTime_offAxis_UR(self.RRs[:,ii], self.TTs[:,ii], self.Betas[:,ii], alpha[ii])
                RRs = self.RRs[:,ii]

                filTM  = where(tts<=max(ttobs))[0]
                filTm  = where(tts[filTM]>=min(ttobs))[0]
                """
                #print(len(tts[filT]))
                #Rint = interp1d(ttobs, self.RRs)
                #Gamint = interp1d(self.RRs, self.Gams[:,ii])
                """
                Rint = interp1d(ttobs, RRs)
                Gamint = interp1d(RRs, self.Gams[:,ii])
                Robs = Rint(tts[filTM][filTm])
                #onAxisTint = interp1d(self.RRs, self.TTs[:,ii])
                #onAxisTobs = onAxisTint(Robs)
                GamObs = Gamint(Robs)
                BetaObs = sqrt(1.-GamObs**(-2.))
                if len(GamObs)==0: continue
                #onAxisTobs = obsTime_onAxis_integrated(Robs, GamObs, BetaObs)
                onAxisTint = interp1d(RRs, self.TTs[:,ii])
                #onAxisTobs = obsTime_onAxis_adiabatic(Robs, BetaObs)
                onAxisTobs  = onAxisTint(Robs)
                """

                ttobs = obsTime_offAxis_UR(self.RRs[:,ii], self.TTs[:,ii], self.Betas[:,ii], alpha[ii])
                RRs = self.RRs[:,ii]

                filTM  = where(tts<=max(ttobs))[0]
                filTm  = where(tts[filTM]>=min(ttobs))[0]

                Rint = interp1d(ttobs, RRs)
                Gamint = interp1d(RRs, self.Gams[:,ii])
                Robs = Rint(tts[filTM][filTm])
                GamObs = Gamint(Robs)
                BetaObs = sqrt(1.-GamObs**(-2.))

                dopFacs =  self.dopplerFactor(calpha[ii], sqrt(1.-GamObs**(-2)))
                afac = self.cellSize/maximum(self.cellSize*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))

                onAxisTobs = dopFacs*tts[filTM][filTm]


                # Forward shock stuff
                Bfield = sqrt(32.*pi*self.nn*self.epB*cts.mp)*cts.cc*GamObs
                gamMobs, nuMobs = minGam(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield)
                gamCobs, nuCobs = critGam(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, onAxisTobs)
                Fnuobs = fluxMax(Robs, GamObs, self.nn, Bfield, self.DD)
                #Reverse shock stuff
                nuM_RS, nuC_RS, Fnu_RS = self.params_tt_RS(onAxisTobs, ii, Rb)


                dopFacs =  self.dopplerFactor(calpha[ii], sqrt(1.-GamObs**(-2)))
                afac = self.cellSize/maximum(self.cellSize*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))

                for freq in obsFreqs:
                    fil1, fil2 = where(gamMobs<=gamCobs)[0], where(gamMobs>gamCobs)[0]
                    fil3, fil4 = where(nuM_RS<=nuC_RS)[0], where(nuM_RS>nuC_RS)[0]
                    freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                    #print shape(freqs), shape(freqs[fil1]), shape(nuMobs[fil1]), shape(nuCobs[fil1]), shape(Fnuobs[fil1]), shape(afac[fil1]), shape(calpha)
                    #print shape(light_curve[obsFreqs==freq, filT]), shape([fil1])
                    #print fil1
                    light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                                    afac[fil1] * dopFacs[fil1]**3. * FluxNuSC_arr(self, nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))*calpha[ii]

                    #light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                    #                                afac[fil2] * dopFacs[fil2]**3. * FluxNuFC_arr(self, nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))*calpha[ii]
                    light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] + (
                                                    afac[fil3] * dopFacs[fil3]**3. * FluxNuSC_arr(self, nuM_RS[fil3], nuC_RS[fil3], Fnu_RS[fil3], freqs[fil3]))*calpha[ii]
                    #light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] + (
                    #                                afac[fil4] * dopFacs[fil4]**3. * FluxNuFC_arr(self, nuM_RS[fil4], nuC_RS[fil4], Fnu_RS[fil4], freqs[fil4]))*calpha[ii]
                    #cont1 = afac[fil1] * dopFacs[fil1]**3. * self.FluxNuSC_arr(nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1])*calpha[ii]
                    #cont2 = afac[fil2] * dopFacs[fil2]**3. * self.FluxNuFC_arr(nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2])*calpha[ii]

                    #light_curve[obsFreqs==freq, filT][fil1] += cont1
                    #light_curve[obsFreqs==freq, filT][fil2] += cont2


            return tts, light_curve, light_curve_RS
            #return tts, 2.*light_curve, 2.*light_curve_RS

        def light_curve_peer(self, theta_obs, obsFreqs, tt0, ttf, num, Rb):

            if type(obsFreqs)==float:
                obsFreqs  = array([obsFreqs])

            calpha = self.obsangle(theta_obs)
            alpha  = arccos(calpha)


            Tfil = self.TTs[:,-1]== max(self.TTs[:,-1])

            max_Tobs = max(obsTime_offAxis_General(self.RRs, self.TTs[:,-1], max(alpha)))

            if ttf>max_Tobs:
                print "ttf larger than maximum observable time. Adjusting value."
                ttf = max_Tobs

            lt0 = log10(tt0*cts.sTd) # Convert to seconds and then logspace
            ltf = log10(ttf*cts.sTd) # Convert to seconds and then logspace

            tts = logspace(lt0, ltf+(ltf-lt0)/num, num) # Timeline on which the flux is evaluated.

            light_curve    = zeros([len(obsFreqs), num])
            light_curve_RS = zeros([len(obsFreqs), num])

            for ii in tqdm(range(self.ncells)):
                ttobs = obsTime_offAxis_General(self.RRs, self.TTs[:,ii], alpha[ii])
                RRs = self.RRs


                filTM  = where(tts<=max(ttobs))[0]
                filTm  = where(tts[filTM]>=min(ttobs))[0]
                #print(len(tts[filT]))
                #Rint = interp1d(ttobs, self.RRs)
                #Gamint = interp1d(self.RRs, self.Gams[:,ii])
                Rint = interp1d(ttobs, RRs)
                Gamint = interp1d(RRs, self.Gams[:,ii])
                Robs = Rint(tts[filTM][filTm])
                #onAxisTint = interp1d(self.RRs, self.TTs[:,ii])
                #onAxisTobs = onAxisTint(Robs)
                GamObs = Gamint(Robs)
                BetaObs = sqrt(1.-GamObs**(-2.))
                if len(GamObs)==0: continue
                onAxisTint = interp1d(RRs, self.TTs[:,ii])
                #onAxisTobs = obsTime_onAxis_integrated(Robs, GamObs, BetaObs)
                onAxisTobs  = onAxisTint(Robs)
                #Bfield = sqrt(32.*pi*cts.mp*self.nn*self.epB*GamObs*(GamObs-1.))*cts.cc
                #gamMobs, nuMobs = minGam_modified(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield)
                #gamCobs, nuCobs = critGam_modified(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, onAxisTobs)

                Bfield = Bfield_modified(GamObs, BetaObs, self.nn, self.epB)
                gamMobs, nuMobs = minGam_modified(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, self.Xp)
                gamCobs, nuCobs = critGam_modified(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, onAxisTobs)
                #nuMobs, nuCobs  = GamObs*nuMobs, GamObs*nuCobs
                Fnuobs = fluxMax_modified(Robs, GamObs, self.nn, Bfield, self.DD, self.PhiP)



                #Reverse shock stuff
                nuM_RS, nuC_RS, Fnu_RS = self.params_tt_RS(onAxisTobs, ii, Rb)




                dopFacs =  self.dopplerFactor(calpha[ii], sqrt(1.-GamObs**(-2)))
                #afac = self.cellSize/maximum(self.cellSize*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))

                for freq in obsFreqs:
                    fil1, fil2 = where(gamMobs<=gamCobs)[0], where(gamMobs>gamCobs)[0]
                    fil3, fil4 = where(nuM_RS<=nuC_RS)[0], where(nuM_RS>nuC_RS)[0]
                    freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                    #print shape(freqs), shape(freqs[fil1]), shape(nuMobs[fil1]), shape(nuCobs[fil1]), shape(Fnuobs[fil1]), shape(afac[fil1]), shape(calpha)
                    #print shape(light_curve[obsFreqs==freq, filT]), shape([fil1])
                    #print fil1
                    light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                                self.cellSize*(GamObs[fil1]*(1.-BetaObs[fil1]*calpha[ii]))**(-3.) * FluxNuSC_arr(self, nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))#*calpha[ii]
                    #light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                    #                            (GamObs[fil2]*(1.-BetaObs[fil2]*calpha[fil2][ii]))**(-3.) * FluxNuFC_arr(self, nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))#*calpha[ii]
                    light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] + (
                                                self.cellSize*(GamObs[fil3]*(1.-BetaObs[fil3]*calpha[ii]))**(-3.) * FluxNuSC_arr(self, nuM_RS[fil3], nuC_RS[fil3], Fnu_RS[fil3], freqs[fil3]))#*calpha[ii]
                    #light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] + (
                    #                            (GamObs[fil4]*(1.-BetaObs[fil4]*calpha[fil4][ii]))**(-3.)* FluxNuFC_arr(self, nuM_RS[fil4], nuC_RS[fil4], Fnu_RS[fil4], freqs[fil4]))#*calpha[ii]


            return tts, light_curve, light_curve_RS
            #return tts, 2.*light_curve, 2.*light_curve_RS


        def lightCurve_interp(self, theta_obs, obsFreqs, tt0, ttf, num, Rb):
            if self.evolution == "adiabatic":
                tts, light_curve, light_curve_RS = self.light_curve_adiabatic(theta_obs, obsFreqs, tt0, ttf, num, Rb)
            elif self.evolution == "peer":
                tts, light_curve, light_curve_RS = self.light_curve_peer(theta_obs, obsFreqs, tt0, ttf, num, Rb)

            return tts, light_curve, light_curve_RS

        def skymap(self, theta_obs, tt_obs, freq, nx, ny, xx0, yy0):

            calpha = self.obsangle(theta_obs)
            alpha  = arccos(calpha)

            TTs, RRs, Gams, Betas = zeros(self.ncells), zeros(self.ncells), zeros(self.ncells), zeros(self.ncells)
            nuMs, nuCs, fluxes    = zeros(self.ncells), zeros(self.ncells), zeros(self.ncells)

            im_xxs = -1.*cos(theta_obs)*sin(self.cthetas)*sin(self.cphis) + sin(theta_obs)*cos(self.cthetas)
            im_yys = sin(self.cthetas)*cos(self.cphis)

            if self.evolution == 'adiabatic':
                for ii in tqdm(range(self.ncells)):
                    Tint = interp1d(self.RRs, self.TTs[:,ii])
                    ttobs = obsTime_offAxis_UR(self.RRs, self.TTs[:,ii], self.Betas[:,ii], alpha[ii])
                    Rint = interp1d(ttobs, self.RRs)
                    RRs[ii] = Rint(tt_obs)
                    TTs[ii] = Tint(RRs[ii])

                    GamInt   = interp1d(self.RRs, self.Gams[:,ii])
                    Gams[ii] = GamInt(RRs[ii])
                    Betas[ii] = sqrt(1.-Gams[ii]**(-2.))

                Bf        = (32.*pi*self.nn*self.epB*cts.mp)**(1./2.) * Gams*cts.cc
                gamM, nuM = minGam(Gams, self.epE, self.epB, self.nn, self.pp, Bf)
                gamC, nuC = critGam(Gams, self.epE, self.epB, self.nn, self.pp, Bf, TTs)
                fluxMax   = fluxMax(RRs, Gams, self.nn, Bf, self.DD)
                #fluxMax[Gams<=2] = 0.

                dopFacs =  self.dopplerFactor(calpha, sqrt(1.-Gams**(-2)))
                afac = self.cellSize/maximum(self.cellSize*ones(num), 2.*pi*(1.-cos(1./Gams)))
                obsFreqs = freq/dopFacs

                fluxes = (self.DD**2./(calpha*self.cellSize*RRs**2.)) * afac * dopFacs**3. * FluxNuSC_arr(self, nuM, nuC, fluxMax, obsFreqs)*1./calpha

            elif self.evolution == 'peer':
                for ii in tqdm(range(self.ncells)):
                    Tint = interp1d(self.RRs, self.TTs[:,ii])
                    ttobs = obsTime_offAxis_General(self.RRs, self.TTs[:,ii], alpha[ii])
                    Rint = interp1d(ttobs, self.RRs)
                    RRs[ii] = Rint(tt_obs)
                    TTs[ii] = Tint(RRs[ii])

                    GamInt   = interp1d(self.RRs, self.Gams[:,ii])
                    #Gams[ii] = self.GamInt(RRs[ii])
                    Gams[ii] = GamInt(RRs[ii])
                    Betas[ii] = sqrt(1.-Gams[ii]**(-2.))

                Bf        = Bfield_modified(Gams, Betas, self.nn, self.epB)
                gamM, nuM = minGam_modified(Gams, self.epE, self.epB, self.nn, self.pp, Bf, self.Xp)
                gamC, nuC = critGam_modified(Gams, self.epE, self.epB, self.nn, self.pp, Bf, TTs)
                fluxMax   = fluxMax_modified(RRs, Gams, self.nn, Bf, self.DD, self.PhiP)
                #fluxMax[Gams<=5] = 0.
                #nuM, nuC = nuM/Gams, nuC/Gams

                dopFacs =  self.dopplerFactor(calpha, sqrt(1.-Gams**(-2)))
                obsFreqs = freq/dopFacs

                afac = self.cellSize/maximum(self.cellSize*ones(self.ncells), 2.*pi*(1.-cos(1./Gams)))
                fluxes = (self.DD**2./(calpha*self.cellSize*RRs**2.)) *self.cellSize*  (Gams*(1.-Betas*calpha))**(-3.) * FluxNuSC_arr(self, nuM, nuC, fluxMax, obsFreqs)
                #fluxes = (Gams*(1.-Betas*calpha))**(-3.) * FluxNuSC_arr(self, nuM, nuC, fluxMax, obsFreqs)*1./calpha
                fluxes2 = self.cellSize*(Gams*(1.-Betas*calpha))**(-3.)*FluxNuSC_arr(self, nuM, nuC, fluxMax, obsFreqs)




            im_xxs = RRs*im_xxs
            im_yys = RRs*im_yys

            return im_xxs, im_yys, fluxes, fluxes2, RRs, Gams, calpha, TTs
        """
        def skymap_2(self, theta_obs, tt_obs, freq, nx, ny, xx0, yy0):

            calpha = self.obsangle(theta_obs)
            alpha  = arccos(calpha)

            TTs, RRs, Gams, Betas = zeros(self.ncells), zeros(self.ncells), zeros(self.ncells), zeros(self.ncells)
            nuMs, nuCs, fluxes    = zeros(self.ncells), zeros(self.ncells), zeros(self.ncells)
            im_xxs = sin(self.cthetas)*cos(self.cphis)
            #im_yys = (1.-sin(alpha))*cos(alpha)*sin(self.cthetas)*sin(self.cphis) - (1.-cos(alpha))*sin(alpha)*cos(self.cthetas)
            im_yys = cos(alpha)*sin(self.cthetas)*sin(self.cphis) - sin(alpha)*cos(self.cthetas)

            if self.evolution == 'adiabatic':
                for ii in tqdm(range(self.ncells)):
                    Tint = interp1d(self.TTs[:,ii], self.RRs)
                    ttobs = obsTime_offAxis_UR(self.RRs, self.TTs[:,ii], self.Betas[:,ii], alpha[ii])
                    Rint = interp1d(ttobs, self.RRs)
                    RRs[ii] = Rint(tt_obs)
                    TTs[ii] = Tint(RRs[ii])

                    GamInt   = interp1d(self.RRs, self.Gams[:,ii])
                    Gams[ii] = GamInt(Robs)
                    Betas[ii] = sqrt(1.-GamObs**(-2.))

                Bf        = (32.*pi*self.nn*self.epB*cts.mp)**(1./2.) * Gams*cts.cc
                gamM, nuM = minGam(Gams, self.epE, self.epB, self.nn, self.pp, Bf)
                gamC, nuC = critGam(Gams, self.epE, self.epB, self.nn, self.pp, Bf, TTs)
                fluxMax   = fluxMax(RRs, Gams, self.nn, Bf, self.DD)

                dopFacs =  self.dopplerFactor(calpha, sqrt(1.-Gams**(-2)))
                afac = self.cellSize/maximum(self.cellSize*ones(num), 2.*pi*(1.-cos(1./Gams)))
                obsFreqs = freq/dopFacs

                fluxes = afac * dopFacs**3. * FluxNuSC_arr(self, nuM, nuC, fluxMax, obsFreqs)*calpha

            elif self.evolution == 'peer':
                for ii in tqdm(range(self.ncells)):
                    Tint = interp1d(self.RRs, self.TTs[:,ii])
                    ttobs = obsTime_offAxis_General(self.RRs, self.TTs[:,ii], alpha[ii])
                    Rint = interp1d(ttobs, self.RRs)
                    RRs[ii] = Rint(tt_obs)
                    TTs[ii] = Tint(RRs[ii])

                    GamInt   = interp1d(self.RRs, self.Gams[:,ii])
                    Gams[ii] = self.GamInt(RRs[ii])
                    Betas[ii] = sqrt(1.-Gams[ii]**(-2.))

                Bf        = Bfield_modified(Gams, Betas, self.nn, self.epB)
                gamM, nuM = minGam_modified(Gams, self.epE, self.epB, self.nn, self.pp, Bf, self.Xp)
                gamC, nuC = critGam_modified(Gams, self.epE, self.epB, self.nn, self.pp, Bf, TTs)
                fluxMax   = fluxMax_modified(RRs, Gams, self.nn, Bf, self.DD, self.PhiP)

                dopFacs =  self.dopplerFactor(calpha, sqrt(1.-Gams**(-2)))
                obsFreqs = freq/dopFacs

                fluxes = self.cellSize * (Gams*(1.-Betas*calpha))**(-3.) * FluxNuSC_arr(self, nuM, nuC, fluxMax, obsFreqs)/calpha



            im_xxs = RRs*im_xxs
            im_yys = RRs*(im_yys + sin(alpha))

            return im_xxs, im_yys, fluxes, RRs, Gams, calpha
        """
        """
            if type(obsFreqs)==float:
                obsFreqs  = array([obsFreqs])

            calpha = self.obsangle(theta_obs)
            alpha  = arccos(calpha)


            Tfil = self.TTs[:,-1]== max(self.TTs[:,-1])
            if self.evolution == 'peer':
                max_Tobs = max(obsTime_offAxis_General(self.RRs, self.TTs[:,-1], max(alpha)))
            elif self.evolution == 'adiabatic':
                max_Tobs = self.RRs[Tfil, -1]/(self.Betas[Tfil,-1]*cts.cc) * (1.-self.Betas[Tfil,-1]*cos(max(alpha)))
            #max_Tobs_oa = max(self.TTs[:,-1])
            #max_Tobs = max(obsTime_offAxis(self, self.RRs, self.TTs[:,alpha==max(alpha)][:,0], max(alpha)))/cts.sTd
            if ttf>max_Tobs:
                print "ttf larger than maximum observable time. Adjusting value. "
                ttf = max_Tobs

            lt0 = log10(tt0*cts.sTd) # Convert to seconds and then logspace
            ltf = log10(ttf*cts.sTd) # Convert to seconds and then logspace

            tts = logspace(lt0, ltf+(ltf-lt0)/num, num) # Timeline on which the flux is evaluated.


            light_curve    = zeros([len(obsFreqs), num])
            light_curve_RS = zeros([len(obsFreqs), num])

            for ii in tqdm(range(self.ncells)):
            #for ii in range(self.ncells):
                if self.evolution == 'peer':
                    ttobs = obsTime_offAxis_General(self.RRs, self.TTs[:,ii], alpha[ii])
                    RRs = self.RRs
                elif self.evolution == 'adiabatic':
                    ttobs = obsTime_offAxis_UR(self.RRs[:,ii], self.TTs[:,ii], self.Betas[:,ii], alpha[ii])
                    RRs = self.RRs[:,ii]

                filTM  = where(tts<=max(ttobs))[0]
                filTm  = where(tts[filTM]>=min(ttobs))[0]
                #print(len(tts[filT]))
                #Rint = interp1d(ttobs, self.RRs)
                #Gamint = interp1d(self.RRs, self.Gams[:,ii])
                Rint = interp1d(ttobs, RRs)
                Gamint = interp1d(RRs, self.Gams[:,ii])
                Robs = Rint(tts[filTM][filTm])
                #onAxisTint = interp1d(self.RRs, self.TTs[:,ii])
                #onAxisTobs = onAxisTint(Robs)
                GamObs = Gamint(Robs)
                BetaObs = sqrt(1.-GamObs**(-2.))
                if len(GamObs)==0: continue
                #onAxisTobs = obsTime_onAxis_integrated(Robs, GamObs, BetaObs)
                if self.evolution == 'adiabatic':
                    onAxisTobs = obsTime_onAxis_adiabatic(Robs, BetaObs)
                elif self.evolution == 'peer':
                    onAxisTobs = obsTime_onAxis_integrated(Robs, GamObs, BetaObs)

                #gamMobs, gamCobs = self.gamMI(Robs), self.gamCI(Robs)
                #nuMobs, nuCobs   = self.nuMI(Robs), self.nuCI(Robs)

                # Forward shock stuff
                if self.evolution == 'adiabatic':
                    Bfield = sqrt(32.*pi*self.nn*self.epB*cts.mp)*cts.cc*GamObs
                    gamMobs, nuMobs = minGam(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield)
                    gamCobs, nuCobs = critGam(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, onAxisTobs)
                elif self.evolution == 'peer':
                    #Bfield = sqrt(32.*pi*cts.mp*self.nn*self.epB*GamObs*(GamObs-1.))*cts.cc
                    #gamMobs, nuMobs = minGam_modified(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield)
                    #gamCobs, nuCobs = critGam_modified(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, onAxisTobs)

                    Bfield = Bfield_modified(GamObs, BetaObs, self.nn, self.epB)
                    gamMobs, nuMobs = minGam_modified(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, self.Xp)
                    gamCobs, nuCobs = critGam_modified(GamObs, self.epE, self.epB, self.nn, self.pp, Bfield, onAxisTobs)
                    #nuMobs, nuCobs  = GamObs*nuMobs, GamObs*nuCobs
                    Fnuobs = fluxMax_modified(Robs, GamObs, self.nn, Bfield, self.DD, self.PhiP)


                #Fnuobs = self.nn**(3./2.)*Robs**3. * cts.sigT * cts.cc**3. *cts.me* (32.*pi*cts.mp*self.epB
                #         )**(1./2.)*GamObs**2./(9.*cts.qe*self.DD**2.)
                Fnuobs = fluxMax(Robs, GamObs, self.nn, Bfield, self.DD)
                #Reverse shock stuff
                nuM_RS, nuC_RS, Fnu_RS = self.params_tt_RS(onAxisTobs, ii, Rb)



                dopFacs =  self.dopplerFactor(calpha[ii], sqrt(1.-GamObs**(-2)))
                afac = self.cellSize/maximum(self.cellSize*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))

                for freq in obsFreqs:
                    fil1, fil2 = where(gamMobs<=gamCobs)[0], where(gamMobs>gamCobs)[0]
                    fil3, fil4 = where(nuM_RS<=nuC_RS)[0], where(nuM_RS>nuC_RS)[0]
                    freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                    #print shape(freqs), shape(freqs[fil1]), shape(nuMobs[fil1]), shape(nuCobs[fil1]), shape(Fnuobs[fil1]), shape(afac[fil1]), shape(calpha)
                    #print shape(light_curve[obsFreqs==freq, filT]), shape([fil1])
                    #print fil1
                    light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                                    afac[fil1] * dopFacs[fil1]**3. * FluxNuSC_arr(self, nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))*calpha[ii]
                    light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                                                    afac[fil2] * dopFacs[fil2]**3. * FluxNuFC_arr(self, nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))*calpha[ii]
                    light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] + (
                                                    afac[fil3] * dopFacs[fil3]**3. * FluxNuSC_arr(self, nuM_RS[fil3], nuC_RS[fil3], Fnu_RS[fil3], freqs[fil3]))*calpha[ii]
                    light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] + (
                                                    afac[fil4] * dopFacs[fil4]**3. * FluxNuFC_arr(self, nuM_RS[fil4], nuC_RS[fil4], Fnu_RS[fil4], freqs[fil4]))*calpha[ii]
                    #cont1 = afac[fil1] * dopFacs[fil1]**3. * self.FluxNuSC_arr(nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1])*calpha[ii]
                    #cont2 = afac[fil2] * dopFacs[fil2]**3. * self.FluxNuFC_arr(nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2])*calpha[ii]

                    #light_curve[obsFreqs==freq, filT][fil1] += cont1
                    #light_curve[obsFreqs==freq, filT][fil2] += cont2

            """

"""

        def lightCurve_interp_adiabatic(self, theta_obs, obsFreqs, tt0, ttf, num):


            if type(obsFreqs)==float:
                obsFreqs  = array([obsFreqs])

            calpha = self.obsangle(theta_obs)
            alpha  = arccos(calpha)

            Tfil = self.TTs[:,-1]== max(self.TTs[:,-1])
            max_Tobs = self.RRs[Tfil,-1]/(self.Betas[Tfil,-1]*cts.cc) * (1.-self.Betas[Tfil,-1]*cos(max(alpha)))
            #max_Tobs_oa = max(self.TTs[:,-1])
            #max_Tobs = max(obsTime_offAxis(self, self.RRs, self.TTs[:,alpha==max(alpha)][:,0], max(alpha)))/cts.sTd
            if ttf>max_Tobs:
                print "ttf larger than maximum observable time. Adjusting value. "
                ttf = max_Tobs

            lt0 = log10(tt0*cts.sTd) # Convert to seconds and then logspace
            ltf = log10(ttf*cts.sTd) # Convert to seconds and then logspace

            tts = logspace(lt0, ltf+(ltf-lt0)/num, num) # Timeline on which the flux is evaluated.


            light_curve    = zeros([len(obsFreqs), num])
            light_curve_RS = zeros([len(obsFreqs), num])

            for ii in tqdm(range(self.ncells)):
            #for ii in range(self.ncells):
                ttobs = obsTime_offAxis(self.RRs[:,ii], self.TTs[:,ii], self.Betas[:,ii], alpha[ii])
                filTM  = where(tts<=max(ttobs))[0]
                filTm  = where(tts[filTM]>=min(ttobs))[0]
                #print(len(tts[filT]))
                Rint = interp1d(ttobs, self.RRs[:,ii])
                Gamint = interp1d(self.RRs[:,ii], self.Gams[:,ii])
                Robs = Rint(tts[filTM][filTm])
                #onAxisTint = interp1d(self.RRs, self.TTs[:,ii])
                #onAxisTobs = onAxisTint(Robs)
                GamObs = Gamint(Robs)
                BetaObs = sqrt(1.-GamObs**(-2.))
                onAxisTobs = obsTime_onAxis_adiabatic(Robs, BetaObs)
                #gamMobs, gamCobs = self.gamMI(Robs), self.gamCI(Robs)
                #nuMobs, nuCobs   = self.nuMI(Robs), self.nuCI(Robs)

                # Forward shock stuff
                gamMobs = self.epE*(self.pp-2.)/(self.pp-1.) * cts.mp/cts.me * GamObs
                gamCobs = 3.*cts.me/(16.*self.epB*cts.sigT*cts.mp*cts.cc*GamObs**3.*onAxisTobs*self.nn)
                nuMobs = GamObs*gamMobs**2.*cts.qe*(32.*pi*cts.mp*self.epB*self.nn)**(1./2.)*GamObs*cts.cc/(2.*pi*cts.me*cts.cc)
                nuCobs = GamObs*gamCobs**2.*cts.qe*(32.*pi*cts.mp*self.epB*self.nn)**(1./2.)*GamObs*cts.cc/(2.*pi*cts.me*cts.cc)
                Fnuobs = self.nn**(3./2.)*Robs**3. * cts.sigT * cts.cc**3. *cts.me* (32.*pi*cts.mp*self.epB
                         )**(1./2.)*GamObs**2./(9.*cts.qe*self.DD**2.)

                #Reverse shock stuff
                nuM_RS, nuC_RS, Fnu_RS = params_tt_RS(self, onAxisTobs, Rb)


                dopFacs =  self.dopplerFactor(calpha[ii], sqrt(1.-GamObs**(-2)))
                afac = self.cellSize/maximum(self.cellSize*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))

                for freq in obsFreqs:
                    fil1, fil2 = where(gamMobs<=gamCobs)[0], where(gamMobs>gamCobs)[0]
                    fil3, fil4 = where(nuM_RS<=nuC_RS)[0], where(nuM_RS>nuC_RS)[0]
                    freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                    #print shape(freqs), shape(freqs[fil1]), shape(nuMobs[fil1]), shape(nuCobs[fil1]), shape(Fnuobs[fil1]), shape(afac[fil1]), shape(calpha)
                    #print shape(light_curve[obsFreqs==freq, filT]), shape([fil1])
                    #print fil1
                    light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                                    afac[fil1] * dopFacs[fil1]**3. * FluxNuSC_arr(self, nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))*calpha[ii]
                    light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                                                    afac[fil2] * dopFacs[fil2]**3. * FluxNuFC_arr(self, nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))*calpha[ii]
                    light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil3]] + (
                                                    afac[fil3] * dopFacs[fil3]**3. * FluxNuFC_arr(self, nuM_RS[fil3], nuC_RS[fil3], Fnu_RS[fil3], freqs[fil3]))*calpha[ii]
                    light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] = light_curve_RS[obsFreqs==freq, filTM[filTm][fil4]] + (
                                                    afac[fil4] * dopFacs[fil4]**3. * FluxNuFC_arr(self, nuM_RS[fil4], nuC_RS[fil4], Fnu_RS[fil4], freqs[fil4]))*calpha[ii]
                    #cont1 = afac[fil1] * dopFacs[fil1]**3. * self.FluxNuSC_arr(nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1])*calpha[ii]
                    #cont2 = afac[fil2] * dopFacs[fil2]**3. * self.FluxNuFC_arr(nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2])*calpha[ii]

                    #light_curve[obsFreqs==freq, filT][fil1] += cont1
                    #light_curve[obsFreqs==freq, filT][fil2] += cont2


            return tts, light_curve, light_curve_RS
"""

"""
    def lightCurve(self, theta_obs, obsFreqs, tt0, ttf, num):
"""
#Generate light curve using histogram
"""

        if type(obsFreqs)==float:
            obsFreqs  = array([obsFreqs])
        #if type(theta_obs==float): theta_obs = array([theta_obs])
        lt0 = log10(tt0*cts.sTd) # Convert to seconds and then logspace
        ltf = log10(ttf*cts.sTd) # Convert to seconds and then logspace
        #tts, fac = self.timeBins(tt0, ttf, num)
        tts = logspace(lt0-(ltf-lt0)/num, ltf+(ltf-lt0)/num, num+1)
        binSize = diff(tts)
        #tts = 10**tts
        Ttot = zeros(num)

        calpha = self.obsangle(theta_obs)
        alpha  = arccos(calpha)

        light_curve   = zeros([len(obsFreqs), num])
        counts        = zeros([len(obsFreqs), num])
        flux_seg      = zeros(len(alpha))


        obsTimeNow = obsTime_offAxis(self, self.RRs[0], self.TTs[0], alpha)#/cts.sTd
        obsTimeNext = obsTime_offAxis(self, self.RRs[1], self.TTs[1], alpha)#/cts.sTd

        #afac = 1.

        for ii in tqdm(range(self.steps-1)):
        #for ii in range(self.steps-1):
            dopFacs = self.dopplerFactor(calpha, self.Betas[ii])
            #obsTimeNow = self.obsTime_offAxis(self.RRs[ii], self.TTs[ii], alpha)#/cts.sTd

            binnedT, bins, tbindex = binned_statistic(obsTimeNow, obsTimeNow, bins=tts, statistic='count')
            tfac = minimum((tts[tbindex]-(obsTimeNow)), (obsTimeNext-obsTimeNow))#/cts.sTd)/df[ii-1] #(tts[tbindex]-tts[tbindex-1])
            fil1   = obsTimeNext>tts[tbindex]
            tfac1  = obsTimeNext[fil1] - tts[tbindex][fil1]
            #Ttot = Ttot + array([((tts[xx]-obsTime[tbindex==xx])).sum() for xx in range(1,len(tts))])
            #Ttot = Ttot + array(map(lambda xx: (tts[xx]-obsTime[tbindex==xx]).sum(), range(1,len(tts))))
            Ttot  = Ttot + array( [tfac[tbindex==xx].sum() + tfac1[tbindex[fil1]==xx].sum() for xx in range(1, len(tts)) ])#array( [(minimum((tts[xx]-obsTimeNow[tbindex==xx]), (obsTimeNext[tbindex==xx]-obsTimeNow[tbindex==xx]))).sum()
                                #         for xx in range(1, len(tts))]  )
            #Ttot = Ttot + array([tfac1[tbindex[fil1]==xx].sum() for xx in range(1, len(tts))])

            #Ttot = Ttot + array([(tts[ii] - obsTime[(obsTime<=tts[ii-1]) & (obsTime>tts[ii])]).sum() for ii in range(1,len(tts))])

            afac = self.cellSize/max(self.cellSize, 2.*pi*(1.-cos(1./self.Gams[ii])))
            #print afac,self.cellSize, 2.*pi*(1.-cos(1./self.Gams[ii]))
            for freq in obsFreqs:
                freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                if self.gM[ii] <= self.gC[ii]:         # Calculate fluxes
                    flux_seg = afac * dopFacs**3. * FluxNuSC(self, self.nuM[ii], self.nuC[ii], self.FnuMax[ii], freqs)*calpha*tfac
                else:
                    flux_seg = afac * dopFacs**3. * FluxNuFC(self, self.nuM[ii], self.nuC[ii], self.FnuMax[ii], freqs)*calpha*tfac
                #dopfacs[ii]


                #count, bins, ind = binned_statistic(obsTimeNow, flux_seg, bins=tts, statistic='count')
                fluxes = array([flux_seg[tbindex==xx].sum() for xx in range(1,len(tts)) ])

                fluxes[tbindex[fil1]] = fluxes[tbindex[fil1]] + flux_seg[fil1]*tfac1

                # Add overflow

                #ncounts = array([ ind[ind==xx].sum() for  xx in range(1,len(tts)) ])
                #ncounts[ncounts==0] = 1.
                light_curve[obsFreqs==freq, :] = light_curve[obsFreqs==freq, :] + fluxes
                #counts[obsFreqs==freq, :]      = counts[obsFreqs==freq, :] + count
            obsTimeNow[:] = obsTimeNext[:]
            obsTimeNext   = obsTime_offAxis(self, self.RRs[ii+1], self.TTs[ii+1], alpha)
            #light_curve = light_curve/self.ncells

        # Return mid-points of the time bins and the light curve
        return tts, light_curve, Ttot

"""
