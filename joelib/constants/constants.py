# Physical, mathematical and unit conversion constants
from numpy import pi

# CGS gaussian units

# Physical constants
GG   = 6.67408E-11            # SI units
cc   = 2.99792E10              # cm/s
mp   = 1.6726219E-24          # proton mass in g
me   = 9.10938356E-28         # electron mass in g
qe   = 4.80320427E-10         # Electron charge in statocoulombs
#qe   = 1.60217662E-19         # absolute value of electron charge in coulombs
#mu0  = 4.E-7 *pi              # vacuum permeability in Henry/meter
sigT = 6.624e-25               # Thompson scatering cross section, cgs units
#sigT = 8./3. *pi * (qe**2./(4.*pi*ep0*cc**2.*me))**2.  # Thomson scattering cross section

# Astronomy specific constants

Msun = 1.98855E33             # kg
au   = 14959787070000         # Astronomical units in cm
pc   = 3.0857e18              # Parsec in cm

# Conversion factors

sTy  = 365.*24.*60.*60.       # seconds to years conversion factor
sTd  = 24.*60.*60.            # seconds to days conversion factor

"""
# SI units

# Physical constants
GG   = 6.67408E-11            # SI units
cc   = 2.99792E8              # m/s
mp   = 1.6726219E-27          # proton mass in kg
me   = 9.10938356E-31         # electron mass in kg
qe   = 1.60217662E-19         # absolute value of electron charge in coulombs
ep0  = 8.8541878E-12          # vacuum permitivity in faradays meter^-1
mu0  = 4.E-7 *pi              # vacuum permeability in Henry/meter
sigT = 8./3. *pi * (qe**2./(4.*pi*ep0*cc**2.*me))**2.  # Thomson scattering cross section


"""
