import astropy.units as u
import numpy as np
from astropy.constants import m_p, G, c, sigma_T, k_B, sigma_sb, R_sun, M_sun
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


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
    return G * M  / (c ** 2 * R)


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


def Q_rad(H, R, M):
    """Everything in CGS"""
    w = keplerian_angular_w(R, M)
    Qrad = Mdotprime(H, R) * R * w**2 /(8 * np.pi)
    return Qrad


def estimate_rho(Mdot, H, w, alpha=0.1):
    """This assumes the radial velocity can be estimated using the
    formula given in the text after ADVECTIVE DISKS WITH MASS LOSS. Returns
    the density (which has a radial dependency)
    All parameters must be given in cgs

    Parameters
    ----------
    Mdot: float,
        Mass-transfer rate at the given radius
    H: float
        Scale height at the given radius
    w:float
        Keplerian angular velocity at the given radius
    """
    return Mdot / (4 * np.pi * alpha * H**3 * w)


def dW_dr(W, R, M, mdot0=1000, ewind=0.5):
    Medd = (eddington_luminosity(M * u.g).decompose(bases=[u.cm, u.g, u.s]) / (c.to(u.cm/u.s))** 2 / 0.1).value
    def mdot_r(r, mdot0, ewind=ewind):
        misco = mass_transfer_inner_radius(mdot0, ewind) * mdot0
        Rsph = 1.62 * mdot0 * (10**6 * u.cm).value
        m_r = misco + (mdot0 - misco) * r / Rsph
        return m_r
    return 2 / (R) * W - (1 / (4 * np.pi * (R)**2) * np.sqrt(G.decompose(bases=u.cgs.bases).value * M / (R)))  * mdot_r(R, mdot0) * Medd


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


def Mdotprime(H, R):
    """From equation 12 and replacing Qrad using Equation 8 and the Pressure using Equation 9
    All parameters in cgs units

    Parameters
    ----------
    H:float,
        Scale height
    R: float,
        Radius
    """
    return 8 * np.pi * c * H / (k_T * R) # correct units

def rhoPrime(rho, Mdot, H, R, M, alpha=0.1):
    """Derivative of the density (unused for now)"""
    w = keplerian_angular_w(R, M)
    return Mdot / (4 * np.pi * alpha * R * H**3 * w) - 2 * np.pi * R*alpha*w * H* rho**2 / Mdot - 21 * rho / R - 4/3 * np.pi * c * R * rho / (Mdot *H * k_T)


def Hprime(Mdot, H, R, M, alpha=0.1):
    """Derivative of the scale height
    This was derived using Qrad (given above) and Qadv = -3/4wWrphi (Equation 22 and 23)
    and replacing P with equation (9)
    """
    w = keplerian_angular_w(R, M)
    rho = estimate_rho(Mdot, H, w, alpha)
    return  2/3 * np.pi *R* alpha* w *H**2 * rho / (Mdot) -4/9 * np.pi * R* c / (Mdot * k_T) - 2 * H /R + Mdot  / (24 * rho* np.pi * R * alpha * w * H**2)


def solve(R, y, M=1.4 * M_sun.to(u.g).value, alpha=0.1, R0=1, Rsph=100, mdot=100):
    #rho, H, Mdot = variables[0]
    #rho = variables[0]
    Mdot = y[0]
    H = y[1]
    # FOR SOME WEIRD REASON SOLVE_BVP CHANGES THE SIZE OF Mdot and H, so we need to redo the grid values
    R = np.linspace(1, Rsph, len(Mdot)) * R0
    #rho_ = rhoPrime(rho, Mdot, H, R, M, alpha)
    Mdot_ = Mdotprime(H, R)
    H_ = Hprime(Mdot, H, R, M, alpha)
    return [Mdot_, H_]


def boundary_conditions(ya, yb, Mdot0=0):
    """ya and yb are the values of the variables (Mdot and H in this case) at the boundary values (Rin and Rsph)"""
    Mdot_a, H_a = ya
    Mdot_b, H_b = yb
    return [Mdot_b - Mdot0, H_a - 10**(-8)]

def main():
    M = (10 * u.M_sun).to(u.g).value
    R0 = isco_radius(M)
    mdot = 200
    Mdot_0 = eddington_accretion_rate(M  / (M_sun.to(u.g).value), R0) * mdot
    ewind = 1
    # In theory we need to estimate the spherizaiton radius. For now we use the value from Poutanen+2007 (Equation 21)
    Rsph = (1.34 - 0.4 * ewind + 0.1 *ewind**2 -(1.1 - ewind * 0.7) * mdot ** (-2/3)) * mdot # in units of R0
    Rsph = 500
    R = np.arange(1, Rsph, 0.01) * R0

    alpha = 0.1
    H_R = 0.6 # the scale height typically hovers around 0.6-0.8
    H_guess = H_R * R
    H_guess[0] = 10**(-8) # boundary condition, H=0 at the inner radius
    #rho_guess = Mdot_0.value * 0.6 / (4 * np.pi * R**3)
    #rho_guess[0] = 0
    # Assume the linear scaling from SS73 to start with
    Mdot_guess = Mdot_0 * R / (Rsph * R0)
    Mdot_guess[-1] = Mdot_0 # boundary condition, Mdot_0 at the Rsph

    y = np.array([Mdot_guess, H_guess])
    plt.figure()
    plt.plot(R / R0, y[1] / R, label="H/R")
    plt.xscale("log")
    plt.plot(R/R0, y[0] / Mdot_0, label="$\dot{M}$ / M$_0$")
    #plt.plot(R/R0, solution.y[0] / (Mdot_0 / R**3), label="$\\rho$ / (M$_0$ / R$^3$)")
    plt.legend()
    plt.xlabel("$R/R_0$")
    plt.savefig("todelete_init.png")
    solution = solve_bvp(lambda x,y: solve(R, y, M=M, alpha=alpha, R0=R0, mdot=mdot, Rsph=Rsph),
                             lambda y0, y1: boundary_conditions(y0, y1, Mdot0=Mdot_0), R, y, verbose=2,
                         max_nodes=1000000)
    R = solution.x
    plt.figure()
    plt.plot(R / R0, solution.y[1] / R, label="H/R")
    plt.xscale("log")
    plt.plot(R/R0, solution.y[0] / Mdot_0, label="$\dot{M}$ / M$_0$")
    #plt.plot(R/R0, solution.y[0] / (Mdot_0 / R**3), label="$\\rho$ / (M$_0$ / R$^3$)")
    plt.legend()
    plt.xlabel("$R/R_0$")
    plt.margins(x=0)
    plt.savefig("todelete.png")

    plt.figure()
    plt.plot(R / R0, Q_rad(solution.y[1] * R**2, R, M), label="Q$_{rad}\\times R^2$")
    plt.xscale("log")
    #plt.yscale("log")
    #plt.plot(R/R0, solution.y[0] / (Mdot_0 / R**3), label="$\\rho$ / (M$_0$ / R$^3$)")
    plt.legend()
    plt.xlabel("$R/R_0$")
    plt.margins(x=0)
    plt.savefig("todelete2.png")


if __name__ == "__main__":
    main()
