from numpy import *
import joelib.constants.constants as cts
from synchrotron_afterglow import afterglow, adiabatic_afterglow


class jetHeadUD(adiabatic_afterglow):


###############################################################################################
# Methods for initializing the cells in the jet head
###############################################################################################

    def __init__(self, EE,  Gam0, nn, epE, epB, pp,DD, nlayers, joAngle):#, obsAngle=0.0):

        adiabatic_afterglow.__init__(self, EE,  Gam0, nn, epE, epB, pp, DD)
        self.nlayers  = nlayers                         # Number of layers for the partition
        #self.nn1      = nn1                            # Number of cells in the first layer
        self.__totalCells()                             # obtain self.ncells
        self.joAngle  = joAngle                         # Jet opening angle
        #self.obsAngle = obsAngle                        # Angle of jet axis with respect to line of sight
        self.angExt   = 2.*pi*(1.-cos(joAngle/2.))      # Solid area covered by the jet head
        #self.cellSize = angsize/self.ncells            # Angular size of each cell
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

    def timeBins(self, tt0, ttf, num):
        """
        Generate time bins for light curves. Time in days.
        The bin array is generated with two buffer bins on either side for photons
        arriving before tt0 or after ttf.
        """
        lt0, ltf = log10(tt0), log10(ttf)
        fac = (ltf-lt0)/num
        lbins = arange(lt0-fac, ltf+fac, fac)

        return lbins, fac

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

    def dopplerFactor(self, cosa):
        """
        Calculate the doppler factors of the different jethead segments
        cosa -> cosine of observeration angle, obtained using obsangle
        """

        beta = (1.-1./self.Gam**2.)**(1./2.)

        return (1.-beta)/(1.-beta*cosa)



    def lightCurve(self, theta_obs, obsFreqs, tt0, ttf, num):
        """
        Generate the light curve at timepoints in tbins
        """

        # Lines for handling multiple observed frequencies

        if type(obsFreqs==float): obsFreqs=array([obsFreqs])
        calpha = self.obsangle(theta_obs)
        alpha  = arccos(calpha)

        lt0 = log10(tt0)
        ltf = log10(ttf)
        tts, fac = self.timeBins(tt0, ttf, num)

        light_curve   = zeros([len(obsFreqs), num+2])
        flux_seg = zeros(len(alpha))

        # Evolution is in terms of R. Obtain the radii correspnding to tt0 and
        # ttf to evaluate the evolution.

        RRs = self.onAxisR(cts.sTd*10**(tts))

        # Evolution and evaluation block
        for RR in RRs:
            print RR, self.Gam
            if(self.Gam <= 1.5): break
            self.updateAG(RR)
            ttO = log10(self.obsTime(alpha)/cts.sTd)               # Calculate the time at which photons from each segment will be observed
            #ttO = ttO - tts[0]
            tindex = floor((ttO-tts[0])/fac).astype('int')  # Find out where to bin the flux values
            tindex[tindex>num] = num

            dopFactor = self.dopplerFactor(calpha)       # Calculate Doppler factors for current instant in evolution

            for freq in obsFreqs:
                freqs = freq/dopFactor              # Calculate the rest-frame frequencies correspondng to the observed frequency
                if self.GamMin <= self.GamCrit:         # Calculate fluxes
                    flux_seg = (self.angExt/(4.*pi)) * dopFactor**3. * self.FluxNuSC(freqs)*calpha
                else:
                    flux_seg = (self.angExt/(4.*pi)) * dopFactor**3. * self.FluxNuFC(freqs)*calpha

                for ii in range(num+2):
                    #print("Filling in"), ii, shape(tindex[tindex==ii]), sum(flux_seg[tindex==ii])
                    light_curve[obsFreqs==freq, ii] = light_curve[obsFreqs==freq, ii] +  sum(flux_seg[tindex==ii])
        return tts, light_curve
