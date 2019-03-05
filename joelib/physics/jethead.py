from numpy import *
import joelib.constants.constants as cts
from synchrotron_afterglow import afterglow


class jetCell(afterglow):

    def __init__(self, EE,  Gam0, nn, epE, epB, pp, agtype, DD, rr, phi):
        afterglow.__init__(self, EE,  Gam0, nn, epE, epB, pp, agtype, DD)
        self.rr = rr                            # Distance from centre
        self.phi = phi                          # Angular position



class jetHeadUD(jetCell):

    def __init__(self, EE,  Gam0, nn, epE, epB, pp, jetType, DD, type, nlayers, nn1, angsize):
        self.jetType  = jetType                 # Type of jet
        self.nlayers  = nlayers                 # Number of layers for the partition
        self.nn1      = nn1                     # Number of cells in the first layer
        self.ncells   = nn1*nlayers*nlayers     # Number of cells
        self.angsize  = angsize                 # Angular size of the jet
        self.cellSize = angsize/ncells          # Angular size of each cell
        self.cells    = self.__makeCells()
        self.ee       = EE/ncells

###############################################################################################

###############################################################################################


    def __makeCells(self):
        self.cells = []
        for ii in range(nlayers):            # Loop over layers
            num = cellsInLayer(self, ii)
            rr = (ii + 0.5)/nlayers          # Midpoint of the cell, scaled to unit radius.
            for jj in range(num):            # Loop over cells in the layer
                phi = (2*jj+1)*pi/num        # Angular coordinate of the midpoint of the cell
                self.cells.append(           # Generate the cell
                    jetCell.__init__(self.ee,  Gam0, nn, epE, epB, pp, agtype, DD, rr, phi))


###############################################################################################

###############################################################################################



    def cellsInLayer(self, ii):
        return (2*ii-1)*self.nn1

    def getcell(self, layer):
        return

    def lightCurve(self, obsangle):
        return
