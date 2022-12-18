import astropy.units as u
import numpy as np
from astropy.constants import m_p, G, c, sigma_T, k_B, sigma_sb, R_sun, M_sun
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import os
import warnings


k_T = (sigma_T / m_p).decompose(bases=u.cgs.bases).value
c = c.decompose(bases=u.cgs.bases).value
G = G.decompose(bases=u.cgs.bases).value
m_p = m_p.to(u.g).value


def eddington_luminosity(M):
    """The classical Eddington luminosity.
    Parameters
    ----------
    M: float
        In solar masses
    Returns the Eddington luminosity in erg/s
    """
    Ledd = 4 * np.pi * G *M *c / k_T
    return Ledd


def accretion_efficiency(M, R):
    """Returns the accretion efficiency.
    M: float
        Mass of the compact object in g
    R: float
        Radius of the compact object or innermost stable orbit
    Returns the accretion efficiency (dimensionless quantity)
    """
    return G * M  / (2* c ** 2 * R)


def eddington_accretion_rate(M, R_in):
    """The classical Eddington luminosity for a given mass.
    Parameters
    ----------
    M: astropy.quantity
    R_in: astropy.quantity
    Returns the Eddington accretion rate in quantity
    """
    efficiency = accretion_efficiency(M, R_in)
    # convert erg to cgs
    return eddington_luminosity(M) / efficiency / c**2


def scale_height(m_r, R, M, spin=0):
    """Equation 18 from Lipunova+99, works for both sub and super critical disks as long as advection is neglected
        Just replace Mdot(R) by the appropiate calculation (i.e. without or with outflows)
        Everything in cgs units
    m_r:float
        (Dimensionless) Mass-transfer rate at every radii
    R: float
        Radii at which the scale height is to be calculated
    M: float
        Mass of the compact object
    """
    R0 = isco_radius(M, spin)
    Rs = 2 * gravitational_radius(M)
    efficiency = accretion_efficiency(M, R0)
    H = Rs * m_r * 3 / 4 / efficiency * (1 - np.sqrt(R0/R))
    return H


def keplerian_angular_w(R, M):
    """Calulate the Keplerian angular velocity at a given radius
    In cgs
    R: float,
        Radius at which to calculate the velocity in cm
    M: float,
        Mass of the compact object in grams
    """
    return np.sqrt(G* M /  R**3)


def gravitational_radius(M):
    """Returns the scharzchild radius for a given mass in km.
    Parameters
    ----------
    M: float,
        Mass of the compact object in grams
    """
    return G * M / (c**2)


def mass_transfer_inner_radius(m_0, e_wind=0.5):
    """Equation 23 from Poutanen et al 2007

        Parameters
        ----------
        m_0: float
            Mass-transfer rate at the donor
        e_wind: float
            Fraction of radiative energy that goes to accelerate the outflow
        Returns the mass-transfer rate at the inner radius of the disk in units of m_0 (i.e. fraction of m_0 that makes it to the BH or NS)
        """
    a = e_wind * (0.83 - 0.25 * e_wind)

    return (1 - a) / (1 - a * (2/5 * m_0) ** (- 0.5) )


def isco_radius(M, a=0.998):
    """Returns the ISCO radius for a given mass in km.
    Parameters
    ----------
    M: float,
        Mass of the compact objects in grams
    a: float,
        Dimensionless spin: 0 for a Scharzschild black hole or 0.998 for a Kerr black hole."""
    z1 = 1 + (1 - a**2) ** (1/3) * ((1 + a)** (1/3) + (1-a) ** (1/3))
    z2 = np.sqrt(3 * a ** 2 + z1**2)
    return (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2))) * gravitational_radius(M)


def Q_rad_shakura(w, R0, Mdot, R):
    return 3 / (8 * np.pi) * w**2 * Mdot * (1 - np.sqrt(R0 / R))


def find_rsph(Qrad, radii, M, spin=0):
    """Find the spherization radius by imposing 4pi int[Qradrdr] = LEdd
    radii:float
        In cm
    """
    R0 = isco_radius(M, spin)
    Ledd = eddington_luminosity(M)
    Rmax = radii[-1]
    R_a = radii[-1] - 1
    # La - Ledd < 0
    R_b = radii[0]
    # Lb - Ledd > 0
    err_tol = R0 * 10 ** (-6)
    err = (R_a - R_b) / 2
    delta_r = np.median(np.diff(radii))

    Ltot = 4 * np.pi * np.sum(Qrad * radii) * (delta_r)
    if Ltot < Ledd:
        warnings.warn("Spherization radius is ill defined as the disk does not exceed its Eddington limit. Increase radii to be explored")
        return radii[0] * 0.9, Ltot
    while (err > err_tol):
        R_c = (R_a + R_b) / 2
        r_range = (radii < Rmax) & (radii > R_c)
        r = radii[r_range]
        Qrad_range = Qrad[r_range]
        L_c = 4 * np.pi * np.sum(Qrad_range * r) * (delta_r)
        fLedd_c = L_c - Ledd
        r_range = (radii < Rmax)  & (radii > R_a)
        r = radii[r_range]
        Qrad_range = Qrad[r_range]
        L_a = 4 * np.pi * np.sum(Qrad_range * r) * (delta_r) # bigger than Eddington
        fLedd_a = L_a - Ledd
        if fLedd_a * fLedd_c >0:
            R_a = R_c
        else:
            R_b = R_c
        err = (R_a - R_b) / 2
    return R_c, L_c

class Disk():
    def __init__(self, M, mdot, alpha=0.1, spin=0, name="disk", Rmin=1, Rmax=500):
        self.M = M
        self.mdot = mdot
        self.alpha = alpha
        self.spin = spin
        self.R0 = isco_radius(M, spin)
        self.Rg = gravitational_radius(M)
        self.Mdot_0 = eddington_accretion_rate(self.M  / (M_sun.to(u.g).value), self.R0) * self.mdot
        self.name = name
        print("Disk %s with M = %.1f M_sun, dot(m) = %.1f and alpha = %.1f and spin = %.1f" % (self.name, self.M / M_sun.to(u.g).value,
                      self.mdot, self.alpha, self.spin))
        self.Rmin = Rmin
        self.Rmax = Rmax


    def density(self, Wrphi, H, w):
        """The sign must be flipped to get positive density
        Parameters
        ----------
        Wrphi: float
            The torque
        H: float
            Scale height
        w: float
            Keplerian angular velocity
        """
        return -Wrphi / (2 * self.alpha * w**2 * H**3)


    def Q_rad(self, H, R):
        """Radiative energy per unit surface. All quantities in cgs
        Parameters
        ----------
        H: float,
            Scale height
        R: float
            Radius
        """
        w = keplerian_angular_w(R, self.M)
        Qrad = H * w**2 *c /k_T
        return Qrad # checked


    def v_r(self, Mdot, H, rho, R):
        """Radial velocity. Equation 13 from Lipunova+1999"""
        return Mdot / (rho * H * 4 * np.pi * R)


class ShakuraSunyaevDisk(Disk):
    def __init__(self, M, mdot, alpha=0.1, torque_Rmin=0, spin=0, Rmin=1, Rmax=500, name="ShakuraSunyaevDisk"):
        super().__init__(M, mdot, alpha=alpha, spin=spin, name=name, Rmin=Rmin, Rmax=Rmax)
        self.torque_Rmin = torque_Rmin
        self.Rsph = 1.62 * self.mdot

    def H(self, R, w):
        """Disk height. Inputs in cgs units

        Parameters
        ----------
        R: float
        w:float
        """
        Wrphi = self.torque(R, w)
        H = - 3 / 4 * k_T * Wrphi / (w * c)
        return H

    def torque(self, R, w):
        """Analytical expression for the torque when Mass loss is conserved
        and assuming the torque at the inner boundary is 0
        """
        Rmin = self.Rmin * self.R0
        return -(self.Mdot_0 * w / (2 * np.pi) * (1 - np.sqrt(Rmin / R)) + self.torque_Rmin * (Rmin / R)**2)


    def torque_derivative(self, R, w):
        """Derivative of the Torque"""
        Rmin = self.Rmin * self.R0
        return -(self.Mdot_0 * w / (4 * np.pi * R) * (4 * np.sqrt(Rmin / R) - 3) - 2 * self.torque_Rmin * (Rmin)**2 / (R**3))


class DiskWithOutflows(Disk):
    def __init__(self, M, mdot, alpha=0.1, spin=0, ewind=1, Rmin=1, Rmax=500, name="with_outflows"):
        super().__init__(M, mdot, alpha=alpha, spin=spin, name=name, Rmin=Rmin, Rmax=Rmax)
        self.ewind = 1
        # Approximative solution for Rpsh based on Poutanen+2007
        self.Rsph = 1.62 * self.mdot # in units of R0

    def H(self, Wrphi, w):

        return -3/4 * Wrphi * k_T / (w * c)

    def torque_derivative(self, Mdot, Wrphi, R, w):
        """Derivative of the torque
        Derived from Equation (4)
        Parameters
        ----------
        Mdot: float,
            Mass-transfer rate at a given radius
        Wrphi: float
            The value of the torque at a given radius
        R: float
            The radius
        w: float
            Keplerian angular velocity at the given radius
        """
        return -(Mdot * w / (4 * np.pi)  + 2 * Wrphi) / R


    def Mdotprime(self, H, R):
        """From equation 12 and replacing Qrad using Equation 8 and the Pressure using Equation 9
        All parameters in cgs units
        Parameters
        ----------
        H:float,
            Scale height
        R: float,
            Radius
        """
        return 8 * np.pi * c * H / (k_T * R) # correct units checked


    def boundary_conditions(ya, yb, Mdot0=1):
        """ya and yb are the values of the variables (H, Mdot and Wrphi in this case) at the boundary values (Rmin and Rmax)"""
        Mdot_a, Wrphi_a = ya
        Mdot_b, Wrphi_b = yb
        zero = 10**(-3)
        return [Mdot_b - Mdot0, Wrphi_a - zero]


    def disk_equations(R, y, disk=None):
        Mdot = y[0]
        Wrphi = y[1]
        w = keplerian_angular_w(R, disk.M)
        dWrphi = disk.torque_derivative(Mdot, Wrphi, R, w)
        H = disk.H(Wrphi, w)
        return [disk.Mdotprime(H, R), dWrphi]


    def solve(self, step=0.05, max_nodes=8000000, **kwargs):
        R = np.arange(self.Rmin, self.Rmax, step) * self.R0
        w = keplerian_angular_w(R, self.M)
        # Assume the linear scaling from SS73 to start with
        M_isco = mass_transfer_inner_radius(self.mdot, self.ewind) * self.Mdot_0
        Mdot_guess = M_isco + (self.Mdot_0 - M_isco)  * (R / self.R0) / self.Rsph  # boundary condition, Mdot_0 at the Rsph
        Wrphi_guess = -self.Mdot_0 * R / (self.Rsph * self.R0) * w / (4 * np.pi) * (1 - (R / self.R0)**(5/2)) / (1 + 3/2 * (R / self.R0)**(5/2))
        Wrphi_guess[0] = 10**(-10)

        y_a = np.array([Mdot_guess, -Wrphi_guess])

        solution = solve_bvp(lambda R,y: DiskWithOutflows.disk_equations(R, y, self),
                             lambda y0, y1: DiskWithOutflows.boundary_conditions(y0, y1, self.Mdot_0), R, y_a,
                                 max_nodes=max_nodes, **kwargs)
        return solution


class AdvectiveDisk(Disk):

    def Q_adv(self, Mdot, H, dH, rho, drho, R):
        w = keplerian_angular_w(R, self.M)
        factor = 6 * dH * rho - H * drho - 9 * H * rho / R
        return Mdot * w**2 * H / (4 * np.pi * R *  rho) * factor


    def Hprime(self, Mdot, H, R, Wrphi, dWrphi):
        """Derivative of the height of the disk. Everything in cgs units.
        Parameters
        ----------
        Mdot: float,
            Mass-accretion rate at the given radius
        H: float,
            Height of the disk
        Wrphi: float
            Stress tensor in the radial and phi coordinates
        dWrphi: float
            Derivative of the stress tensor
        """
        w = keplerian_angular_w(R, self.M)
        rho = self.density(Wrphi, H, w)
        denominator = 6 * rho - 3/2 * Wrphi / (self.alpha * H**3 * w**2)
        numerator = 9 * H * rho / R - dWrphi / (2 * self.alpha * H**2 *w**2) - 3 /2 * Wrphi / (self.alpha * H**2 * w**2 * R) - 3 * np.pi * R * rho * Wrphi / (Mdot * w * H) - 4*np.pi * R* c * rho /(Mdot * k_T)
        return numerator / denominator


    def Hprime_simplified(self, Mdot, H, R, Wrphi, dWrphi, w):
        """Derivative of the height of the disk. Everything in cgs units. Here rho has been replaced and
        the equations have been greatly simplified (mostly for speed purposes)
        Parameters
        ----------
        Mdot: float,
            Mass-accretion rate at the given radius
        H: float,
            Height of the disk
        Wrphi: float
            Stress tensor in the radial and phi coordinates
        dWrphi: float
            Derivative of the stress tensor
        w: float
            Keplerian angular velocity
        """
        return 1 / 9 * (12 * H / R - 3 * np.pi * R* Wrphi / (w * H * Mdot) + H * dWrphi / Wrphi - 4 * R* np.pi * c /(Mdot * k_T))


    def densityPrime(self, Wrphi, dWrphi, H, dH, Mdot, R):
        """Derivative of the density (unused for now) at the given radius
        Parameters
        ----------
        Wrphi: float,
            Stress tensor
        dWrphi: float,
            Derivative of the stress tensor
        H: float,
            Height of the disk
        dH: float
            Derivative of the height of the disk at the given radius
        Mdot: float,
            Mass-transfer rate at the given radius
        R: float,
            Radius
        """
        w = keplerian_angular_w(R, self.M)
        return -1/(2 * self.alpha * H**3 * w**2) * (dWrphi - 3 * Wrphi * dH / H + 3 * Wrphi / R)


class AdvectiveDiskWithOutflows(AdvectiveDisk):
    def __init__(self, M, mdot, alpha=0.1, spin=0, ewind=1, Rmin=1, Rmax=500, name="advective_with_outflows"):
        super().__init__(M, mdot, alpha=alpha, spin=spin, name=name, Rmin=Rmin, Rmax=Rmax)
        self.ewind = ewind
        # Approximative solution for Rpsh based on Poutanen+2007
        self.Rsph = (1.34 - 0.4 * ewind + 0.1 * ewind**2 -(1.1 - 0.7 * ewind) * (self.mdot ** (-2/3))) * self.mdot # in units of R0


    def boundary_conditions(ya, yb, Mdot0=1):
        """ya and yb are the values of the variables (H, Mdot and Wrphi in this case) at the boundary values (Rmin and Rmax)"""
        H_a, Mdot_a, Wrphi_a = ya
        H_b, Mdot_b, Wrphi_b = yb
        zero = 10**(-10)
        return np.array([H_a -zero, Mdot_b - Mdot0, Wrphi_a + zero])


    def disk_equations(R, y, disk=None):
        H = y[0]
        Mdot = y[1]
        Wrphi = y[2]
        w = keplerian_angular_w(R, disk.M)
        dWrphi = disk.torque_derivative(Mdot, Wrphi, R, w)
        return np.vstack((disk.Hprime_simplified(Mdot, H, R, Wrphi, dWrphi, w), disk.Mdotprime(H, R), dWrphi))


    def solve(self, step=0.05, max_nodes=8000000, **kwargs):
        R = np.arange(self.Rmin, self.Rmax, step) * self.R0
        w = keplerian_angular_w(R, self.M)
        # Assume the linear scaling from SS73 to start with
        M_isco = mass_transfer_inner_radius(self.mdot, self.ewind) * self.Mdot_0
        Mdot_guess = M_isco + (self.Mdot_0 - M_isco)  * (R / self.R0) / self.Rsph  # boundary condition, Mdot_0 at the Rsph
        Wrphi_guess = self.Mdot_0 * R / (self.Rsph * self.R0) * w / (4 * np.pi) * (1 - (R / self.R0)**(5/2)) / (1 + 3/2 * (R / self.R0)**(5/2))
        Wrphi_guess[0] = 10**(-10)
        H_guess = -3/4 * Wrphi_guess * k_T / (w * c)


        y_a = np.array([H_guess, Mdot_guess, Wrphi_guess])

        solution = solve_bvp(lambda R,y: AdvectiveDiskWithOutflows.disk_equations(R, y, self),
                             lambda y0, y1: AdvectiveDiskWithOutflows.boundary_conditions(y0, y1, self.Mdot_0), R, y_a,
                                 max_nodes=max_nodes, **kwargs)
        return solution


    def Mdotprime(self, H, R):
        """From equation 12 and replacing Qrad using Equation 8 and the Pressure using Equation 9
        All parameters in cgs units
        Parameters
        ----------
        H:float,
            Scale height
        R: float,
            Radius
        """
        return 8 * np.pi * c * H / (k_T * R) # correct units checked


    def torque_derivative(self, Mdot, Wrphi, R, w):
        """Derivative of the torque
        Derived from Equation (4)
        Parameters
        ----------
        Mdot: float,
            Mass-transfer rate at a given radius
        Wrphi: float
            The value of the torque at a given radius
        R: float
            The radius
        w: float
            Keplerian angular velocity at the given radius
        """
        return -(Mdot * w / (4 * np.pi)  + 2 * Wrphi) / R


class AdvectiveDiskWithoutOutflows(AdvectiveDisk):

    def __init__(self, M, mdot, alpha=0.1, spin=0, torque_Rmin=10**(-12), Rmin=1, Rmax=500, name="advective_no_outflows"):
        super().__init__(M, mdot, alpha=alpha, spin=spin, name=name, Rmin=Rmin, Rmax=Rmax)
        self.torque_Rmin = torque_Rmin


    def boundary_conditions(ya, yb, H_boundary=10**(-15)):
        """ya and yb are the values of the variables at the boundary values (Rin and Rout)
        H is zero at the inner boundary
        """
        H_a = ya
        return H_a - H_boundary


    def disk_equations(R, y, disk=None):
        H = y[0]
        w = keplerian_angular_w(R, disk.M)
        Wrphi = disk.torque(R, w)
        dWrphi = disk.torque_derivative(R, w)
        #rho_ = rhoPrime(rho, Mdot, H, R, M, alpha)
        H_ = disk.Hprime_simplified(disk.Mdot_0, H, R, Wrphi, dWrphi, w)
        return [H_]

    def solve(self, step=0.005, H_boundary=10**(-4), max_nodes=10000000, **kwargs):
        R = np.arange(self.Rmin, self.Rmax, step) * self.R0
        H_guess = scale_height(self.mdot, R, self.M) # solution is independent of initial guess
        H_guess[0] = H_boundary # boundary condition, H=0 at the inner radius

        y = np.array([H_guess])

        solution = solve_bvp(lambda R,y: AdvectiveDiskWithoutOutflows.disk_equations(R, y, disk=self),
                             lambda y0, y1: AdvectiveDiskWithoutOutflows.boundary_conditions(y0, y1, H_boundary=H_boundary), R, y,
                             max_nodes=max_nodes, **kwargs)
        return solution


    def torque(self, R, w):
        """Analytical expression for the torque when Mass loss is conserved
        and assuming the torque at the inner boundary is 0
        """
        Rmin = self.Rmin * self.R0
        return -(self.Mdot_0 * w / (2 * np.pi) * (1 - np.sqrt(Rmin / R)) + self.torque_Rmin * (Rmin / R)**2)


    def torque_derivative(self, R, w):
        """Derivative of the Torque"""
        Rmin = self.Rmin * self.R0
        return -(self.Mdot_0 * w / (4 * np.pi * R) * (4 * np.sqrt(Rmin / R) - 3) - 2 * self.torque_Rmin * Rmin**2 / (R**3))


def test_solve_full_disk():
    outdir = "lipunova/full_disk"
    Ledd = eddington_luminosity(M)
    #outer_disk = AdvectiveDisk_Without_Outflows(M, mdot, alpha)
    Rsph = 300 # in units of R0
    inner_disk = AdvectiveDiskWithOutflows(M, mdot, alpha, Rmin=1, Rmax=Rsph)
    Rmax = 2000
    outer_disk = ShakuraSunyaevDisk(M, mdot, alpha, Rmin=Rsph, Rmax=Rmax)

    print("Init Rsph")
    print(inner_disk.Rsph)
    R0 = inner_disk.R0
    Rsph_2 = 0
    tolerance = 10**-3
    max_iter = 50
    iter = 0

    L = 0
    while (np.abs(L - Ledd) > tolerance) or iter>max_iter:
        Rsph_2 = Rsph
        inner_disk.Rmax = Rsph_2
        sol_inner_disk = inner_disk.solve(max_nodes=1000000, verbose=2)
        r_in = sol_inner_disk.x
        H_outflow = sol_inner_disk.y[0]
        Mdot_inner = sol_inner_disk.y[1]
        Wrphi = sol_inner_disk.y[2]

        H_Rsph = H_outflow[-1]
        Mdot_Rsph = Mdot_inner[-1]
        w_Rsph = keplerian_angular_w(Rsph_2 * R0, inner_disk.M)
        Wrphi_Rsph = Wrphi[-1]
        outer_disk.torque_Rsph = Wrphi_Rsph
        outer_disk.Rmin = Rsph_2
        #sol_outerdisk = outer_disk.solve(H_boundary=H_Rsph, max_nodes=500000)
        r_out = np.arange(Rsph_2, Rmax, 0.01) * inner_disk.R0
        w = keplerian_angular_w(r_out, inner_disk.M)
        H_no_outflow = outer_disk.H(r_out, w)
        Qout = outer_disk.Q_rad(H_no_outflow, r_out)
        #H_no_outflow = sol_outerdisk.y[0]
        #r_out = sol_outerdisk.x
        #Qout = outer_disk.Q_rad(H_no_outflow, r_out)
        Qrad = np.append(inner_disk.Q_rad(H_outflow, r_in), Qout) #
        r = np.append(r_in, r_out)
        #Qrad = outer_disk.Q_rad(H_outflow, r_in)
        Rsph, L = find_rsph(Qout, r_out, M)
        Rsph = Rsph / R0
        print("Rsph:")
        print(Rsph)
        #print(L / Ledd)
        #print(Rsph / outer_disk.R0)
        R = np.append(r_in, r_out)
        r = R  / outer_disk.R0
        H =  np.append(H_outflow, H_no_outflow)
        Mdot = np.append(Mdot_inner, np.ones(len(r_out)) * outer_disk.Mdot_0)
        margin = 700
        plt.figure()
        plt.xscale("log")
        plt.xlabel("$R/R_0$")
        plt.plot(r[margin:], (H / R)[margin:], label="H/R")
        plt.plot(r[margin:], Mdot[margin:] / inner_disk.Mdot_0, label="$\dot{M}$ / M$_0$")
        plt.axvline(Rsph, ls="--", color="black")
        plt.legend()
        plt.margins(x=0.025)
        plt.ylim(bottom=0, top=2)
        plt.savefig("%s/todelete_%d.png" % (outdir, iter))
        iter+=1

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    plt.figure()
    plt.plot(R / inner_disk.R0, H / R, label="H/R")
    plt.xscale("log")
    plt.plot(R/inner_disk.R0, Mdot / inner_disk.Mdot_0, label="$\dot{M}$ / M$_0$")
    plt.axvline(Rsph / inner_disk.R0, ls="--", color="black")
    plt.legend()
    plt.xlabel("$R/R_0$")
    plt.margins(x=0)
    plt.savefig("%s/todelete.png" %outdir)

def test_disk_outflows():

    disk = AdvectiveDiskWithOutflows(M, mdot, alpha, Rmin=1, spin=0)
    disk.Rmax = disk.Rsph # set the maximum radius to the spherization radius where the outflow starts to develop
    step = 0.1
    solution = disk.solve(step=step, max_nodes=4000000, verbose=2) # reduce the step to reduce the computational burden
    R = solution.x
    height = solution.y[0]
    Mdot = solution.y[1]
    Wrphi = solution.y[2]
    w = keplerian_angular_w(R, disk.M)
    outdir = "lipunova/%s_step%.3f" % (disk.name, step)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    fig = plt.figure()
    fig.suptitle("%s" % disk.name)
    r = R / disk.R0
    plt.plot(r, height/ R, label="$H/R$")
    plt.plot(r, Mdot / disk.Mdot_0, label="$\dot{M} / \dot{M_0}$")
    plt.xlabel("$R / R_0$")
    plt.ylim(bottom=-0.1, top=5)
    plt.legend()
    plt.savefig("%s/todelete_H_R.png" % outdir)

    Qrad = disk.Q_rad(height, R)
    #Qrad[-1] = 0 # boundary condition
    fig = plt.figure()
    fig.suptitle("%s" % disk.name)
    Q0 = keplerian_angular_w(disk.R0, disk.M) * disk.Mdot_0 / (8 * np.pi) # just a normalization factor
    plt.plot(r, Qrad * r**2 / Q0, label="$Q_{rad} r^2 / Q_0$")
    plt.ylim(bottom=0, top=10)
    plt.xlabel("$R / R_0$")
    plt.ylabel("$Q_{rad} \\times r^2 / Q_0$")
    plt.savefig("%s/todelete_Qrad.png" % outdir)


    rho = disk.density(Wrphi, height, w)
    dWrphi = disk.torque_derivative(Mdot, Wrphi, R, w)
    dH = disk.Hprime_simplified(Mdot, height, R, Wrphi, dWrphi, w)
    drho = disk.densityPrime(Wrphi, dWrphi, height, dH, Mdot, R)
    Qadv = disk.Q_adv(Mdot, height, dH, rho, drho, R)
    Q = -3/4 * w * Wrphi
    fig = plt.figure()
    fig.suptitle("%s" % disk.name)
    margin = 200
    plt.plot(r[margin:], (Qrad / Q)[margin:], label="$Q_{rad}$ / $Q^+$")
    plt.plot(r[margin:], (Qadv / Q)[margin:], label="$Q_{adv}$ / $Q^+$")
    plt.ylim(bottom=0, top=1.1)
    plt.legend()
    plt.xlabel("$R / R_0$")
    plt.savefig("%s/todelete_Q.png" % outdir)

    plt.figure()
    fig.suptitle("%s" % disk.name)
    plt.plot(r, Wrphi)
    plt.ylabel("W$_{r\phi}$")
    plt.xlabel("$R / R_0$")
    plt.savefig("%s/todelete_Wrphi.png" % outdir)

    # density
    fig = plt.figure()
    fig.suptitle("%s" % disk.name)
    margin = 300
    plt.plot(r[margin:], rho[margin:]) # avoid the singularity at R_0
    plt.margins(x=0.025, y=0.025)
    plt.ylabel("$\\rho$ (g/cm$^3$)")
    plt.xlabel("$R / R_0$")
    plt.savefig("%s/todelete_rho.png" % outdir)


def test_find_rsph():
    R0 = isco_radius(M, 0)
    step = 0.005
    r = np.arange(1, 200000, step) * R0
    w = keplerian_angular_w(r, M)
    Qrad = Q_rad_shakura(w, R0, Mdot_0, r)
    print(Qrad)
    L = 4 * np.pi * np.sum(Qrad * r) * (step * R0) # tested
    #Rsph, L = find_rsph(Qrad, r, M, Mdot_0)
    print("Resulting Rsph:")
    #print(Rsph / R0)
    print("Using SS73 formula")
    print(9/4 * mdot)

    H = scale_height(mdot, r, M)
    Qrad = Q_rad(H, r, M)
    print(Qrad)
    Rsph, L = find_rsph(Qrad, r, M)
    print(L)
    print("Resulting Rsph:")
    print(Rsph / R0)
    print("Using SS73 formula")
    print(9/4 * mdot)


if __name__ == "__main__":

    if not os.path.isdir("lipunova"):
        os.mkdir("lipunova")
    mdot = 10
    alpha = 0.1
    M = 10 * M_sun.to(u.g).value
    R0 = isco_radius(M, 0)
    Mdot_0 = eddington_accretion_rate(M  / (M_sun.to(u.g).value), R0) * mdot

    #test_solve_disk()
    test_disk_outflows()
    #test_solve_full_disk()
    #test_find_rsph()
    #test_disks()



def test_dwdr():
   M = 1 * u.M_sun
   R_0 = (10**6 * u.cm).value
   radii = np.arange(R_0, 10**3 * R_0, R_0) * u.cm
   mdot = 100
   ewind = 0.5
   W_0 = 0* u.g / u.s**2
   s =  dW_dr(W_0, radii[0],M, mdot, ewind)
   print(s)

def testing() -> None:
    M = 1 * u.M_sun
    R_0 = 1 * 10**6
    radii = np.geomspace(R_0, 10**3 * R_0, R_0)
    mdot = 100
    ewind = 0.5
    W_0 = 0 * u.g / u.s**2
    sol = odeint(dW_dr, W_0.value, radii, args=(M.to(u.g).value, mdot, ewind))
    plt.plot(radii, sol )#* 3 * 4 * keplerian_angular_w(radii * u.cm, M, ).value)
    plt.show()
