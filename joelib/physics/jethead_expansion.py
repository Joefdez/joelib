from numpy import *
#import joelib.constants.constants as ctns
from joelib.constants.constants import cc, mp
from joelib.physics.grb_observables import *
from joelib.physics.afterglow_properties import *
from joelib.physics.afterglow_dynamics import *
from scipy.interpolate import interp1d
#from tqdm import tqdm
#from jethead import jetHeadUD, jetHeadGauss



#################################################################################################################################################################
######################### Evolution of the blast wave as given in Pe'er 2012 inckuding a simple description of lateral expansion ################################
#################################################################################################################################################################

def cellsInLayer(ii):
    """
    Return number of cells in layer ii
    """
    return (2*ii+1)


class jetHeadUD():

        def __init__(self, EE,  Gam0, nn, epE, epB, pp, steps, Rmin, Rmax,
                     evolution, nlayers, initJoAngle, aa=-1, shell_type='thin',
                     Rb=1., withSpread=True):

            self.evolution = evolution
            self.nlayers = nlayers
            self.initJoAngle = initJoAngle
            self.__totalCells()
            #self.__get_thetas()
            self.EE = EE
            self.Gam0 = Gam0
            self.Beta0 = sqrt(1.-Gam0**(-2.))
            self.nn = nn
            self.epE, self.epB, self.pp = epE, epB, pp
            self.Rmin, self.Rmax = Rmin, Rmax
            self.aa = aa                                      # Labels lateral expansion model
            self.angExt0 = 2.*pi*(1.-cos(initJoAngle/2.))/self.ncells
            self.steps = steps
            self.shell_type = shell_type
            self.withSpread = withSpread
            self.Xp   = Xint(pp)
            self.PhiP = PhiPint(pp)
            self.__correct_energy()
            self.__shell_evolution()
            self.__shell_division()
            self.__make_layers()
            self.__peakParamsRS()
            #self.__get_thetas()
            #self.__make_layers()
            #self.__totalCells()
            self.__cell_size()



        def __correct_energy(self):                 # Change energy from spherical equivalent (inherited from parent class) to energy of segment
            self.EE = self.EE*self.angExt0/(4.*pi)
            self.MM0 = self.EE/(self.Gam0*cc**2.)     # This mass is the mass of the ejecta


        def __shell_evolution(self):
            self.Rd = (3./self.angExt0 * 1./(cc**2.*mp) *
                            self.EE/(self.nn*self.Gam0**2.))**(1./3.)
            self.Td = self.Rd*(1.-self.Beta0)/(cc*self.Beta0)

            self.RRs = logspace(log10(self.Rmin), log10(self.Rmax), self.steps)
            self.Gams, self.mms = zeros([len(self.RRs)]), zeros([len(self.RRs)])
            #self.thetas, self.angExt = zeros([len(self.RRs)]), zeros([len(self.RRs)])
            #self.thetas[0], self.angExt[0] = self.initJoAngle/.2, 2*pi*(1.-cos(self.initJoAngle/.2))
            self.TTs = zeros([len(self.RRs)])
            if self.aa>=0 :
                if self.evolution == 'peer':
                # self.MMs contains the swept-up ISM mass, not to be confused with the ejecta mass self.MM0
                    #self.Gams, self.Betas, self.joAngle, self.MMs, self.TTs, __ = solver_GP12(
                    #                            self.MM0, self.Gam0, 0., self.initJoAngle/2., self.RRs, self.nn,
                    #                            self.aa, self.steps, self.angExt0, self.ncells, withSpread = self.withSpread)


                    self.Gams, self.Betas, self.joAngle, self.MMs, self.TTs, __ = solver_expanding_shell(
                                                self.MM0, self.Gam0, 0., self.initJoAngle/2., self.RRs, self.nn,
                                                self.aa, self.steps, self.angExt0, self.ncells, self.Rd, withSpread = self.withSpread)

                elif self.evolution == 'BM':
                    self.Gams, self.Betas, self.joAngle, self.MMs, self.TTs, __ = BMsolver_expanding_shell(
                                                self.MM0, self.Gam0, 0., self.initJoAngle/2., self.RRs, self.nn, self.aa, self.steps, self.angExt0, self.ncells)

                self.joAngle = 2*self.joAngle
            else:
                if self.evolution == 'peer':
                    self.Gams, self.Betas, self.MMs, self.TTs = solver_collimated_shell(
                                                    self.MM0, self.Gam0, self.angExt0, self.RRs, self.nn, self.steps)
                elif self.evolution == 'BM':
                    self.Gams, self.Betas, self.MMs, self.TTs = BMsolver_collimated_shell(
                                                    self.MM0, self.Gam0, self.angExt0, self.RRs, self.nn, self.steps)
                #self.angExt = ones([len(self.RRs)])*self.angExt0/self.ncells
                self.joAngle = ones([len(self.RRs)])*self.initJoAngle

            #self.joAngle = self.thetas[-1]

            self.Bfield  = Bfield_modified(self.Gams, self.Betas, self.nn, self.epB)
            self.gM, self.nuM = minGam_modified(self.Gams, self.epE, self.epB, self.nn, self.pp, self.Bfield, self.Xp)
            self.gC, self.nuC = critGam_modified(self.Gams, self.epE, self.epB, self.nn, self.pp, self.Bfield, self.TTs)
            self.FnuMax = fluxMax_modified(self.RRs, self.Gams, self.MMs/mp, self.Bfield, self.PhiP)

            self.GamInt = interp1d(self.RRs, self.Gams)
            self.gamMI, self.gamCI = interp1d(self.RRs, self.gM), interp1d(self.RRs, self.gC)
            self.nuMI, self.nuCI = interp1d(self.RRs, self.nuM), interp1d(self.RRs, self.nuC)
            self.FnuMaxI = interp1d(self.RRs, self.FnuMax)
            self.neI = interp1d(self.RRs, self.MMs/mp)         # Interpolated number of electrons as a function of R for the flux calculation

            #self.angExtI = interp1d(self.RRs, self.angExt)




        def get_thetas(self, joAngle):

            fac = arange(0,self.nlayers+1)/float(self.nlayers)        # Numerical factor for use during execution
            thetas  = 2.*arcsin(fac*sin(joAngle/4.))         # Calculate the propagation angle with respect to jet axis
            cthetas = 0.5*(thetas[1:]+thetas[:-1])

            return thetas, cthetas

        def __shell_division(self):

            thetas = zeros([self.steps, self.nlayers+1])
            cthetas = zeros([self.steps, self.nlayers])
            cthetasI = []

            thetas0, cthetas0 = self.get_thetas(self.initJoAngle)

            for ii in range(self.steps):
                thetas[ii,:], cthetas[ii,:] = self.get_thetas(self.joAngle[ii])
                #cthetas[ii,:] = cthetas0[:] + 0.5*(self.joAngle[ii]-self.initJoAngle)

            for jj in range(self.nlayers):
                cthetasI.append(interp1d(self.RRs, cthetas[:,jj]))

            self.thetas, self.cthetas, self.cthetasI = thetas, cthetas, cthetasI



        def __make_layers(self):

            self.layer = array([])
            self.phis   = array([])
            self.cphis   = array([])
            #self.cthetas0 = array([])

            for ii in range(self.nlayers):                             # Loop over layers and populate the arrays
                num = cellsInLayer(ii)
                self.phis   = append(self.phis, arange(0,num+1)*2.*pi/num)     # Phi value of the edges
                self.layer   = append(self.layer,ones(num)*(ii+1))           # Layer on which the cells are
                #self.cthetas0 = append(self.cthetas0,ones(num)*0.5*(self.thetas0[ii]+self.thetas0[ii+1]))   # Central theta values of the cells
                self.cphis   = append(self.cphis,(arange(0,num)+0.5)*2.*pi/num )    # Central phi values of the cells

                self.layer = self.layer.astype('int')


        def __totalCells(self):
            tot = 0
            for ii in range(0,self.nlayers):
                tot = tot + cellsInLayer(ii)
                #tot = tot + int(round(cellsInLayer(ii)/2))
            self.ncells = tot


        def __cell_size(self):
            self.angExt = 2.*pi*(1.-cos(self.joAngle/2.))/self.ncells


        def __peakParamsRS(self):

            Gam0 = self.Gam0

            # These need to be scaled be the correspinding factor of Rb when calculating light curve

            if self.shell_type=='thin':
                #print("Settig up thin shell")
                #self.RSpeak_nuM = 9.6e14 * epE**2. * epB**(1./2.) * nn**(1./2) * Gam0**2.
                #self.RSpeak_nuC = 4.0e16 * epB**(-3./2.) * EE**(-2./3.) * nn**(-5./6.) * Gam0**(4./3.)
                #self.RSpeak_Fnu = 5.2 * DD**(-2.) * epB**(1./2.) * EE * nn**(1./2.) * Gam0
                self.RSpeak_nuM  = self.nuMI(self.Rd)/(Gam0**2) #* self.Rb**(1./2.)
                self.RSpeak_nuC  = self.nuCI(self.Rd) #* self.Rb**(-3./2.)*
                self.RSpeak_Fnu =  Gam0*self.FnuMaxI(self.Rd)# * self.Rb**(1./2.)*



class jetHeadGauss():

        def __init__(self, EEc0,  Gamc0, nn, epE, epB, pp, steps, Rmin, Rmax,
                     evolution, nlayers, initJoAngle, coAngle, aa, structure='gaussian',
                     kk=0, shell_type='thin', Rb=1., withSpread=True):
            self.nlayers = nlayers
            self.steps = steps
            self.EEc0 = EEc0
            self.Gamc0 = Gamc0
            self.Rmin, self.Rmax = Rmin, Rmax
            self.nlayers = nlayers
            self.coAngle = coAngle
            thetaMax = 2.*sqrt(-2.*self.coAngle**2. * log(1e-8/(self.Gamc0-1.)))
            self.structure = structure
            self.kk = kk
            self.initJoAngle = min(initJoAngle, thetaMax)                          # Make sure that Gamma > 1 throughout the jet
            self.nn = nn
            self.epE, self.epB, self.pp = epE, epB, pp
            self.aa = aa
            self.Xp   = Xint(pp)
            self.PhiP = PhiPint(pp)
            self.shell_type = shell_type
            self.withSpread = withSpread
            self.__totalCells()
            self.angExt0 = 2.*pi*(1.-cos(initJoAngle/2.))/self.ncells
            self.thetas0, self.cthetas0 = self.get_thetas(self.initJoAngle)
            self.__correct_energy()
            self.__energies_and_LF()
            self.__make_layers()
            self.cell_Rds = (3./(4.*pi) * 1./(cc**2.*mp) *
                                self.cell_EEs/(self.nn*self.cell_Gam0s**2.))**(1./3.)
            self.cell_Tds = self.cell_Rds/(cc*self.cell_Beta0s) * (1.-self.cell_Beta0s)
            if self.Rmin>0.01 * self.cell_Rds.min(): self.Rmin = 0.01 * self.cell_Rds.min()


            self.Gams, self.mms = zeros([self.steps, self.nlayers]), zeros([self.steps, self.nlayers])
            self.Betas = zeros([self.steps, self.nlayers])
            self.TTs = zeros([self.steps, self.nlayers])
            self.theta_edges, self.cthetas = zeros([self.steps, self.nlayers]), zeros([self.steps, self.nlayers])
            self.joAngles = zeros([self.steps, self.nlayers])
            self.__shell_evolution()
            #self.__shell_division()
            self.__thetas_interpolation()
            self.__peakParamsRS_struc()

        def __totalCells(self):
            tot = 0
            for ii in range(0,self.nlayers):
                tot = tot + cellsInLayer(ii)
                #tot = tot + int(round(cellsInLayer(ii)/2))
            self.ncells = tot

        """
        def __cell_size(self):
            self.angExt0 = 2.*pi*(1.-cos(self.joAngle/2.))/self.ncells
        """


        def get_thetas_division(self, layer):

            facs = arange(layer, layer+2)/float(self.nlayers)
            #thetas = 2.*arcsin(facs*sin(self.joAngles[:,layer-1]))
            #cthetas = 0.5*(thetas[:,0]+thetas[:,1])
            cthetas = 0.5*(2.*arcsin(facs[0]*sin(self.joAngles[:,layer-1]/2.)) + 2.*arcsin(facs[1]*sin(self.joAngles[:,layer-1]/2.)))

            return cthetas


        def get_thetas(self, joAngle):

            fac = arange(0,self.nlayers+1)/float(self.nlayers)        # Numerical factor for use during execution
            thetas  = 2.*arcsin(fac*sin(joAngle/4.))         # Calculate the propagation angle with respect to jet axis
            cthetas = 0.5*(thetas[1:]+thetas[:-1])

            return thetas, cthetas


        def __shell_division(self):

            thetas = zeros([self.steps, self.nlayers+1])
            cthetas = zeros([self.steps, self.nlayers])
            for ii in range(self.steps):
                cthetas[ii,:] = self.cthetas0 + (self.joAngles[ii,:]-self.initJoAngle)


            #self.thetas0, self.cthetas0 = thetas[0,:], cthetas[0,:]
            self.theta_edges0 = thetas[0,:] # Initial outmost edges of each cell
            #self.thetas, self.cthetas = thetas, cthetas

        def __make_layers(self):

            self.layer = array([])
            self.phis   = array([])
            self.cphis   = array([])
            #self.cthetas0 = array([])

            for ii in range(self.nlayers):                             # Loop over layers and populate the arrays
                num = cellsInLayer(ii)
                self.phis   = append(self.phis, arange(0,num+1)*2.*pi/num)     # Phi value of the edges
                self.layer   = append(self.layer,ones(num)*(ii+1))           # Layer on which the cells are
                #self.cthetas0 = append(self.cthetas0,ones(num)*0.5*(self.thetas0[ii]+self.thetas0[ii+1]))   # Central theta values of the cells
                self.cphis   = append(self.cphis,(arange(0,num)+0.5)*2.*pi/num )    # Central phi values of the cells

                self.layer = self.layer.astype('int')



        def __correct_energy(self):                 # Change energy from spherical equivalent (inherited from parent class) to energy of segment
            self.EEc0 = self.EEc0*self.angExt0/(4.*pi)
            self.MMc0 = self.EEc0/(self.Gamc0*cc**2.)


        def __energies_and_LF(self):

            if self.structure=='gaussian':
                #AngFacs = exp(-1.*self.cthetas**2./(2.*self.coAngle**2.))
                self.cell_EEs = self.EEc0 * exp(-1.*self.cthetas0**2./(self.coAngle**2.))    # Just for texting
                #self.cell_EEs = self.EE * exp(-1.*self.cthetas**2./(self.coAngle**2.))
                #print shape(self.cthetas0)
                self.cell_Gam0s = 1.+(self.Gamc0-1)*exp(-1.*self.cthetas0**2./(2.*self.coAngle**2.))
                #self.cell_Gam0s[self.cell_Gam0s<=1.+1e-6] == 1.+1.e-6
            elif self.structure=='power-law':
                self.cell_EEs   = zeros(self.nlayers)
                self.cell_Gam0s = zeros(self.nlayers)
                self.cell_EEs[self.cthetas0<=self.coAngle] = self.EEc0
                self.cell_Gam0s[self.cthetas0<=self.coAngle] = self.Gamc0
                wings = self.cthetas0>self.coAngle
                self.cell_EEs[wings] = self.EEc0*(self.cthetas0[wings]/self.coAngle)**(-1.*self.kk)
                self.cell_Gam0s[wings] = 1. + (self.Gamc0-1.)*(self.cthetas0[wings]/self.coAngle)**(-1.*self.kk)


            self.cell_Beta0s = sqrt(1.-(self.cell_Gam0s)**(-2.))
            #self.cell_Beta0s[self.cell_Beta0s<=1e-6] = 1.e-6
            self.cell_MM0s = self.cell_EEs/(self.cell_Gam0s*cc**2.)




        def __thetas_interpolation(self):

            cthetasI = []

            for jj in range(self.nlayers):
                cthetasI.append(interp1d(self.RRs, self.cthetas[:,jj]))

            self.cthetasI = cthetasI


        def __shell_evolution(self):
            self.RRs = logspace(log10(self.Rmin), log10(self.Rmax), self.steps)
            self.Gams, self.mms = zeros([len(self.RRs), len(self.cthetas0)]), zeros([len(self.RRs), len(self.cthetas0)])
            self.thetas, self.angExt = zeros([len(self.RRs), len(self.cthetas0)]), zeros([len(self.RRs), len(self.cthetas0)])
            self.TTs = zeros([len(self.RRs), len(self.cthetas0)])
            self.Bfield = []
            self.TTInt = []
            self.GamInt = []
            self.neI = []
            self.angExtI = []
            self.gamMI, self.gamCI, self.nuMI, self.nuCI = [], [], [], []
            self.FnuMax = []
            for ii in range(self.nlayers):
                if self.aa>=0 :
                    #print shape(self.theta_edges0)
                    CIL = cellsInLayer(ii)
                    self.Gams[:,ii], self.Betas[:,ii], self.joAngles[:,ii], self.mms[:,ii], self.TTs[:,ii], self.angExt[:,ii] = solver_expanding_shell(
                                    self.cell_MM0s[ii], self.cell_Gam0s[ii], 0., self.initJoAngle/2., self.RRs, self.nn, self.aa,
                                    self.steps, self.angExt0, self.ncells, self.cell_Rds[ii], withSpread = self.withSpread)

                    #self.Gams[:,ii], self.Betas[:,ii], self.joAngles[:,ii], self.mms[:,ii], self.TTs[:,ii], __ = solver_GP12(
                    #                self.cell_MM0s[ii], self.cell_Gam0s[ii], 0., self.initJoAngle/2., self.RRs, self.nn, self.aa,
                    #                self.steps, self.angExt0, self.ncells, withSpread = self.withSpread)
                    #self.cthetas[:,ii] = self.cthetas0[ii] + 0.5*(self.theta_edges[:,ii] + self.theta_edges0[ii])
                    self.cthetas[:,ii] = self.get_thetas_division(ii)
                    #self.cthetas[:,ii] = self.cthetas0[ii] + 0.5*(self.joAngles[:,ii]-self.initJoAngle)
                else:
                    #print shape(self.cell_Gam0s), shape(self.cell_Gam0s[ii])
                    self.Gams[:,ii], self.Betas[:,ii], self.mms[:,ii], self.TTs[:,ii] = solver_collimated_shell(
                                    self.cell_MM0s[ii], self.cell_Gam0s[ii], self.angExt0, self.RRs, self.nn, self.steps)
                    self.cthetas[:,ii] = self.cthetas0[ii]

                self.TTInt.append(interp1d(self.RRs, self.TTs[:,ii]))
                self.GamInt.append(interp1d(self.RRs, self.Gams[:,ii]))
                #self.angExt = ones([len(self.RRs)])*self.angExt0/self.ncells
                self.neI.append(interp1d(self.RRs, self.mms[:,ii]/mp))         # Interpolated number of electrons as a function of R for the flux calculation
                self.angExtI.append(interp1d(self.RRs, self.angExt[:,ii]))


                Bf = Bfield_modified(self.Gams[:,ii], self.Betas[:,ii], self.nn, self.epB)
                self.Bfield.append(interp1d(self.RRs, Bf))
                gM, fM =  minGam_modified(self.Gams[:,ii], self.epE, self.epB, self.nn, self.pp, Bf, self.Xp)
                self.gamMI.append(interp1d(self.RRs, gM))
                self.nuMI.append(interp1d(self.RRs, fM))
                gC, fC = critGam_modified(self.Gams[:,ii], self.epE, self.epB, self.nn, self.pp, Bf, self.TTs[:,ii])
                self.gamCI.append(interp1d(self.RRs, gC))
                self.nuCI.append(interp1d(self.RRs, fC))
                Fmax = fluxMax_modified(self.RRs, self.Gams[:,ii], self.mms[:,ii]/mp, Bf, self.PhiP)
                self.FnuMax.append(interp1d(self.RRs, Fmax))





        def __peakParamsRS_struc(self):


            RSpeak_nuM_struc = zeros(self.ncells)
            RSpeak_nuC_struc = zeros(self.ncells)
            RSpeak_Fnu_struc = zeros(self.ncells)


            if self.shell_type=='thin':
                #print("Setting up thin shell")
                for ii in range(self.nlayers):
                    #self.RSpeak_nuM = 9.6e14 * epE**2. * epB**(1./2.) * nn**(1./2) * Gam0**2.
                    #self.RSpeak_nuC = 4.0e16 * epB**(-3./2.) * EE**(-2./3.) * nn**(-5./6.) * Gam0**(4./3.)
                    #self.RSpeak_Fnu = 5.2 * DD**(-2.) * epB**(1./2.) * EE * nn**(1./2.) * Gam0

                    Rd, Td  = self.cell_Rds[ii], self.cell_Tds[ii]
                    #print Rd

                        #print shape(self.RRs), shape(self.Gams)
                    GamsInt = interp1d(self.RRs[:], self.Gams[:,ii])
                    Gam0 = GamsInt(Rd)
                    Beta0 = sqrt(1.-Gam0**(-2.))
                    Bf = Bfield_modified(Gam0, Beta0, self.nn, self.epB)
                    gamM, nuM = minGam_modified(Gam0, self.epE, self.epB, self.nn, self.pp, Bf, self.Xp)
                    gamC, nuC = critGam_modified(Gam0, self.epE, self.epB, self.nn, self.pp, Bf, Td)
                    Fnu = fluxMax_modified(Rd, Gam0, self.nn, Bf, self.PhiP)

                    RSpeak_nuM_struc[ii]  = nuM/(Gam0**2)
                    RSpeak_nuC_struc[ii]  = nuC
                    RSpeak_Fnu_struc[ii] =  Gam0*Fnu


            self.RSpeak_nuM_struc = RSpeak_nuM_struc #self.Rb**(1./2.)*RSpeak_nuM_struc
            self.RSpeak_nuC_struc = RSpeak_nuC_struc #self.Rb**(-3./2.)*RSpeak_nuC_struc
            self.RSpeak_Fnu_struc = RSpeak_Fnu_struc #self.Rb**(1./2.)*RSpeak_Fnu_struc
