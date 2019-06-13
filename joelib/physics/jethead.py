from numpy import *
import joelib.constants.constants as cts
from synchrotron_afterglow import afterglow, adiabatic_afterglow
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from tqdm import tqdm



class jetHeadUD(adiabatic_afterglow):


###############################################################################################
# Methods for initializing the cells in the jet head
###############################################################################################

    def __init__(self, EE,  Gam0, nn, epE, epB, pp, DD, steps, evolution, nlayers, joAngle):#, obsAngle=0.0):

        self.nlayers  = nlayers                         # Number of layers for the partition
        #self.nn1      = nn1                            # Number of cells in the first layer
        self.__totalCells()                             # obtain self.ncells
        self.joAngle  = joAngle                         # Jet opening angle
        #self.obsAngle = obsAngle                        # Angle of jet axis with respect to line of sight
        self.angExt   = 2.*pi*(1.-cos(joAngle/2.))      # Solid area covered by the jet head
        self.cellSize = self.angExt/self.ncells            # Angular size of each cell
        self.__makeCells()                              # Generate the cells: calculate the angular positions of the shells
        adiabatic_afterglow.__init__(self, EE,  Gam0, nn, epE, epB, pp, DD, steps, evolution)
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
            self.layer   = append(self.layer,ones(num+1)*(ii+1))            # Layer on which the phi edges are
            self.phis    = append(self.phis, arange(0,num+1)*2.*pi/num)     # Phi value of the edges
            self.cthetas = append(self.cthetas,ones(num)*0.5*(self.thetas[ii]+self.thetas[ii+1]))   # Central theta values
            self.cphis   = append(self.cphis,(arange(0,num)+0.5)*2.*pi/num )    # Central phi values


    def __totalCells(self):
        tot = 0
        for ii in range(0,self.nlayers):
            tot = tot + self.cellsInLayer(ii)
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


    def lightCurve(self, theta_obs, obsFreqs, tt0, ttf, num):
        """
        Generate light curve using histogram
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


        obsTimeNow = self.obsTime_offAxis(self.RRs[0], self.TTs[0], alpha)#/cts.sTd
        obsTimeNext = self.obsTime_offAxis(self.RRs[1], self.TTs[1], alpha)#/cts.sTd

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
                    flux_seg = afac * dopFacs**3. * self.FluxNuSC(self.nuM[ii], self.nuC[ii], self.FnuMax[ii], freqs)*calpha*tfac
                else:
                    flux_seg = afac * dopFacs**3. * self.FluxNuFC(self.nuM[ii], self.nuC[ii], self.FnuMax[ii], freqs)*calpha*tfac
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
            obsTimeNext   = self.obsTime_offAxis(self.RRs[ii+1], self.TTs[ii+1], alpha)
            #light_curve = light_curve/self.ncells

        # Return mid-points of the time bins and the light curve
        return tts, light_curve, Ttot


    def lightCurve_interp(self, theta_obs, obsFreqs, tt0, ttf, num):


        if type(obsFreqs)==float:
            obsFreqs  = array([obsFreqs])

        calpha = self.obsangle(theta_obs)
        alpha  = arccos(calpha)

        max_Tobs = max(self.obsTime_offAxis(self.RRs, self.TTs, max(alpha)))/cts.sTd
        if ttf>max_Tobs:
            print "ttf larger than maximum observable time. Adjusting value. "
            ttf = max_Tobs

        lt0 = log10(tt0*cts.sTd) # Convert to seconds and then logspace
        ltf = log10(ttf*cts.sTd) # Convert to seconds and then logspace

        tts = logspace(lt0, ltf+(ltf-lt0)/num, num) # Timeline on which the flux is evaluated.


        light_curve   = zeros([len(obsFreqs), num])

        for ii in tqdm(range(self.ncells)):
        #for ii in range(self.ncells):
            ttobs = self.obsTime_offAxis(self.RRs, self.TTs, alpha[ii])
            filTM  = where(tts<=max(ttobs))[0]
            filTm  = where(tts[filTM]>=min(ttobs))[0]
            #print(len(tts[filT]))
            Rint = interp1d(ttobs, self.RRs)
            Robs = Rint(tts[filTM][filTm])
            GamObs = self.GamInt(Robs)
            gamMobs, gamCobs = self.gamMI(Robs), self.gamCI(Robs)
            nuMobs, nuCobs   = self.nuMI(Robs), self.nuCI(Robs)
            Fnuobs = self.FnuMI(Robs)
            dopFacs =  self.dopplerFactor(calpha[ii], sqrt(1.-GamObs**(-2)))
            afac = self.cellSize/maximum(self.cellSize*ones(num)[filTM][filTm], 2.*pi*(1.-cos(1./GamObs)))

            for freq in obsFreqs:
                fil1, fil2 = where(gamMobs<=gamCobs)[0], where(gamMobs>gamCobs)[0]
                freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                #print shape(freqs), shape(freqs[fil1]), shape(nuMobs[fil1]), shape(nuCobs[fil1]), shape(Fnuobs[fil1]), shape(afac[fil1]), shape(calpha)
                #print shape(light_curve[obsFreqs==freq, filT]), shape([fil1])
                #print fil1
                light_curve[obsFreqs==freq, filTM[filTm][fil1]] = light_curve[obsFreqs==freq, filTM[filTm][fil1]] + (
                                                    afac[fil1] * dopFacs[fil1]**3. * self.FluxNuSC_arr(nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1]))*calpha[ii]
                light_curve[obsFreqs==freq, filTM[filTm][fil2]] = light_curve[obsFreqs==freq, filTM[filTm][fil2]] + (
                                                    afac[fil2] * dopFacs[fil2]**3. * self.FluxNuFC_arr(nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2]))*calpha[ii]
                #cont1 = afac[fil1] * dopFacs[fil1]**3. * self.FluxNuSC_arr(nuMobs[fil1], nuCobs[fil1], Fnuobs[fil1], freqs[fil1])*calpha[ii]
                #cont2 = afac[fil2] * dopFacs[fil2]**3. * self.FluxNuFC_arr(nuMobs[fil2], nuCobs[fil2], Fnuobs[fil2], freqs[fil2])*calpha[ii]

                #light_curve[obsFreqs==freq, filT][fil1] += cont1
                #light_curve[obsFreqs==freq, filT][fil2] += cont2


        return tts, light_curve



#class jetHeadGauss(adiabatic_afterglow):
