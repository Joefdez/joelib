# Physical, mathematical and unit conversion constants
from numpy import pi

# Physical constants
GG   = 6.67408E-11            # SI units
cc   = 2.99792E8              # m/s
mp   = 1.6726219E-27          # proton mass in kg
me   = 9.10938356E-31         # electron mass in kg
qe   = 1.60217662E-19         # absolute value of electron charge in coulombs
ep0  = 8.8541878E-12          # vacuum permitivity in faradays meter^-1
mu0  = 4.E-7 *pi              # vacuum permeability in Henry/meter
sigT = 8./3. *pi * (qe**2./(4.*pi*ep0*cc**2.*me))  # Thomson scattering cross section

# Astronomy specific constants

Msun = 1.98855E30             # kg
au   = 149597870700           # Astronomical units in metre
pc   = 3.0857e16              # Parsec in metres

# Conversion factors

sTy  = 365.*24.*60.*60.       # seconds to years conversion factor
sTd  = 24.*60.*60.            # seconds to days conversion factor
