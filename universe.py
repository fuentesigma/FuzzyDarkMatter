#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import math
import numpy as np

# SI constants
G = 6.67430e-11
HBAR = 1.054571817e-34
C = 299792458.0
EV = 1.602176634e-19
MSUN = 1.98847e30
KPC = 3.085677581491367e19
MPC = 1.0e3 * KPC
GYR = 1.0e9 * 365.25 * 24.0 * 3600.0

class Universe:
    # Flat LCDM background and the simulation units
    def __init__(self, H0=70.0, Omega_m=0.27, Omega_L=0.73, m_boson_eV=8.0e-23):
        self.Omega_m, self.Omega_L = Omega_m, Omega_L
        self.m = m_boson_eV * EV / C ** 2                        # boson mass [kg]
        self.H0 = H0 * 1.0e3 / MPC                               # Hubble constant [1/s]
        self.T0 = 1.0 / math.sqrt(1.5 * self.H0 ** 2 * Omega_m)  # code time [s]
        self.L0 = math.sqrt(self.T0 * HBAR / self.m)             # code length [m]
        self.rho_mean = 3.0 * self.H0 ** 2 * Omega_m / (8.0 * math.pi * G)   # mean matter density [kg/m^3]

    def code_length(self, L_Mpc):
        # Comoving length given in Mpc
        return L_Mpc * MPC / self.L0

    def hubble(self, a):
        # H(a) in 1/s (Friedmann equation, flat universe, radiation neglected)
        return self.H0 * math.sqrt(self.Omega_m / a ** 3 + self.Omega_L)

    def da_dtau(self, a):
        # da/dt = a H, and dtau = dt / (a^2 T0) ==> da/dtau = a^3 H T0
        return a ** 3 * self.hubble(a) * self.T0

    def advance(self, a, dtau):
        # a(tau + dtau) it changes < 1 % per substep
        n = max(1, math.ceil(100.0 * dtau * self.da_dtau(a) / a))
        h = dtau / n
        for _ in range(n):
            k1 = self.da_dtau(a)
            k2 = self.da_dtau(a + 0.5 * h * k1)
            k3 = self.da_dtau(a + 0.5 * h * k2)
            k4 = self.da_dtau(a + h * k3)
            a += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return a

    def growing_mode(self, a):
        # In linear theory delta grows as D(a), with f = dlnD/dlna ~ Omega_m(a)^0.55
        # d(delta)/dt = f H delta, and d(delta)/dtau = a^2 T0 f H delta
        f = (self.Omega_m / a ** 3 / (self.hubble(a) / self.H0) ** 2) ** 0.55
        return a ** 2 * self.T0 * f * self.hubble(a)

def load_power_spectrum(filename):
    # Linear matter power spectrum P(k) 
    # The file comes formatted as: k [1/Mpc], P [Mpc^3]
    k, P = np.loadtxt(filename, unpack=True)[:2]
    order = np.argsort(k)
    log_k, log_P = np.log(k[order]), np.log(P[order])

    def P_k(k):
        k = np.asarray(k, dtype=float)
        out = np.zeros_like(k)
        out[k > 0] = np.exp(np.interp(np.log(k[k > 0]), log_k, log_P))
        return out

    return P_k

def initial_wave(n, L_Mpc, universe, P_k, a, seed=1):
    """
    psi = sqrt(rho) exp(iS)

    rho       density
    grad S    velocity (the growing mode of linear theory)
    
    density   rho = 1 + delta, with delta a Gaussian random field of power spectrum P(k)
    velocity  mass conservation
              d(delta)/dtau = -div v
              d(delta)/dtau =  g delta (g = universe.growing_mode(a)) 
              ==> div v = -g delta, where
              v = grad S:  
              lap S = -g delta
              .: S_k = g delta_k / k^2 (in Fourier space)
    """
    rng = np.random.default_rng(seed)

    # 1D fourier frequencies scaled to get wavenumbers [1/Mpc]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L_Mpc / n)

    # Create a 3D grid containing k^2
    k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2

    # Generate complex Gaussian white noise in Fourier space
    noise = rng.normal(size=k2.shape) + 1j * rng.normal(size=k2.shape)
    
    # Scale white noise by P(k) and volume normalisation to get delta_k
    delta_k = noise * np.sqrt(P_k(np.sqrt(k2)) * n ** 3 / L_Mpc ** 3)

    # Clear the DC (zero frequency) component
    delta_k[0, 0, 0] = 0.0

    # Back to real space using an inverse 3D FFT
    delta = np.real(np.fft.ifftn(delta_k, norm="ortho"))
    
    # Rho must stay positive for sqrt(rho)
    delta = np.maximum(delta, -0.999)

    # Use simulation units
    L_code = universe.code_length(L_Mpc)
    k2_sim = k2 * (L_Mpc / L_code) ** 2
    k2_sim[0, 0, 0] = 1.0

    # Solve the Poisson equation in Fourier space to compute the velocity potential S
    S = np.real(np.fft.ifftn(universe.growing_mode(a) * np.fft.fftn(delta) / k2_sim))

    # Construct the complex wave function psi
    psi = np.sqrt(1.0 + delta) * np.exp(1j * S)

    # Normalise the final wave function
    return psi / np.sqrt(np.mean(np.abs(psi) ** 2))