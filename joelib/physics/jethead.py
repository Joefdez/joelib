from numpy import *
import joelib.constants.constants as cts
from synchrotron_afterglow import afterglow, adiabatic_afterglow
from scipy.stats import binned_statistic
from tqdm import tqdm



class jetHeadUD(adiabatic_afterglow):


###############################################################################################
# Methods for initializing the cells in the jet head
###############################################################################################

    def __init__(self, EE,  Gam0, nn, epE, epB, pp, DD, steps, nlayers, joAngle):#, obsAngle=0.0):

        adiabatic_afterglow.__init__(self, EE,  Gam0, nn, epE, epB, pp, DD, steps)
        self.nlayers  = nlayers                         # Number of layers for the partition
        #self.nn1      = nn1                            # Number of cells in the first layer
        self.__totalCells()                             # obtain self.ncells
        self.joAngle  = joAngle                         # Jet opening angle
        #self.obsAngle = obsAngle                        # Angle of jet axis with respect to line of sight
        self.angExt   = 2.*pi*(1.-cos(joAngle/2.))      # Solid area covered by the jet head
        self.cellSize = self.angExt/self.ncells            # Angular size of each cell
        self.ee       = EE/self.ncells                  # Energy per cell
        self.__makeCells()                              # Generate the cells: calculate the angular positions of the shells


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
        #self.delGams   = 2.*pi/(2*arange(0,self.nlayers)+1)        # Calculate the spacing between cells in the plane perpendicular to jet axis (i.e in disk)
        #self.ncellslayer = cellsInLayer(arange(0, self.nlayer))

        for ii in range(self.nlayers):                             # Loop over layers and populate the arrays
            num = self.cellsInLayer(ii)
            self.layer   = append(self.layer,ones(num+1)*(ii+1))            # Layer on which the phi edges are
            self.phis    = append(self.phis, arange(0,num+1)*2.*pi/num)     # Phi value of the edges
            self.cthetas = append(self.cthetas,ones(num)*0.5*(self.thetas[ii]+self.thetas[ii+1]))   # Central theta values
            self.cphis   = append(self.cphis,(arange(0,num)+0.5)*2.*pi/num )    # Central phi values
            #self.cphis   = append(self.cphis, self.delGams[ii]/num *(arange(0,num)+0.5))


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
        Generate light curve
        """

        if type(obsFreqs)==float:
            obsFreqs  = array([obsFreqs])
        #if type(theta_obs==float): theta_obs = array([theta_obs])
        lt0 = log10(tt0*cts.sTd) # Convert to seconds and then logspace
        ltf = log10(ttf*cts.sTd) # Convert to seconds and then logspace
        #tts, fac = self.timeBins(tt0, ttf, num)
        tts = logspace(lt0-(ltf-lt0)/num, ltf+(ltf-lt0)/num, num+1)
        #tts = 10**tts
        Ttot = zeros(num)

        calpha = self.obsangle(theta_obs)
        alpha  = arccos(calpha)

        light_curve   = zeros([len(obsFreqs), num])
        flux_seg = zeros(len(alpha))


        for ii in tqdm(range(self.steps)):
        #for ii in range(self.steps):
            dopFacs = self.dopplerFactor(calpha, self.Betas[ii])
            obsTime = self.obsTime_offAxis(self.RRs[ii], self.TTs[ii], alpha)#/cts.sTd

            binnedT, bins, tbindex = binned_statistic(obsTime, obsTime, bins=tts, statistic='count')
            tfac = (tts[tbindex]-(obsTime))#/cts.sTd)/df[ii-1] #(tts[tbindex]-tts[tbindex-1])
            Ttot = Ttot + array(map(lambda xx: obsTime[tbindex==xx].sum(), range(1,len(tts))))

            for freq in obsFreqs:
                freqs = freq/dopFacs              # Calculate the rest-frame frequencies correspondng to the observed frequency
                if self.gM[ii] <= self.gC[ii]:         # Calculate fluxes
                    flux_seg = (self.cellSize/(4.*pi)) * dopFacs[ii]**3. * self.FluxNuSC(self.nuM[ii], self.nuC[ii], freqs)*calpha*tfac
                else:
                    flux_seg = (self.cellSize/(4.*pi)) * dopFacs[ii]**3. * self.FluxNuFC(self.nuM[ii], self.nuC[ii], freqs)*calpha*tfac


                fluxes, bins, ind = binned_statistic(obsTime, flux_seg, bins=tts, statistic='sum')
                light_curve[obsFreqs==freq, :] = light_curve[obsFreqs==freq, :] + fluxes


            #light_curve = light_curve/self.ncells

        # Return mid-points of the time bins and the light curve
        return 0.5*(tts[:-1]+tts[1:]), light_curve/Ttot
