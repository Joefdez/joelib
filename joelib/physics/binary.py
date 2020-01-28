class binarySystem:

    def __init__(self):
        pass

    def period(m1, m2, aa):

        return 2.*pi*sqrt(aa**3./(GG*(m1+m2)))

    def freq(m1, m2, aa):

        return 1./( 2.*pi*sqrt(aa**3./(GG*(m1+m2))) )

    def sma(m1, m2, TT):

        return (GG*(m1 + m2)/(4.*pi**2) * TT**2.)**(1./3.)

    def binding_energy(mm1, mm2, aa):
        # Binding energy of a binary
        # It is negative
        return -1.*GG*(mm1*mm2)/(2.*aa)



bs = binarysystem()
