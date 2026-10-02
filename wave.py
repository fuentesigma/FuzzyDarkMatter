#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import math
import functools
import torch
import numpy as np

class Box:
    # Periodic cube of side L holding the wave function psi
    def __init__(self, psi, L, four_pi_G=1.0, V_ext=0.0):
        self.psi = torch.as_tensor(psi)
        self.L, self.four_pi_G = L, four_pi_G

        # External potential (a black hole, a host halo), added to self-gravity in the kicks
        self.V_ext = V_ext
        self.n = self.psi.shape[0]
        self.dx = L / self.n
        k = 2.0 * math.pi * torch.fft.fftfreq(self.n, d=self.dx, dtype=torch.float64)
        k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
        self.k = k.to(self.psi.real.dtype).to(self.psi.device)
        self.k2 = k2.to(self.psi.real.dtype).to(self.psi.device)
        self.V = self.potential(self.density())

    def density(self, psi=None):
        return (self.psi if psi is None else psi).abs() ** 2

    def potential(self, rho):
        # Solve lap V = four_pi_G (rho - <rho>)
        rho_k = torch.fft.fftn(rho - rho.mean())
        V_k = -self.four_pi_G * rho_k / torch.where(self.k2 > 0, self.k2, 1.0)
        return torch.fft.ifftn(V_k).real

    def step(self, dt, a=1.0, V_ext_end=None):
        # Advance psi by dt --> kick, drift, kick; V_ext_end is the external potential at the end of the step, if it moves
        psi = self.psi * torch.exp(-1j * a * (self.V + self.V_ext) * dt / 2)
        psi = torch.fft.ifftn(torch.fft.fftn(psi) * torch.exp(-1j * self.k2 * dt / 2))
        self.V = self.potential(self.density(psi))
        if V_ext_end is not None:
            self.V_ext = V_ext_end
        self.psi = psi * torch.exp(-1j * a * (self.V + self.V_ext) * dt / 2)

    def dt_max(self, a=1.0, c_kin=0.25, c_pot=0.5):
        V_max = torch.as_tensor(self.V + self.V_ext).abs().max().item()
        return min(c_kin * self.dx ** 2, c_pot / (a * V_max) if V_max > 0 else math.inf)

    def mass(self):
        return self.density().sum().item() * self.dx ** 3

    def energy(self):
        kinetic = 0.5 * (self.k2 * torch.fft.fftn(self.psi).abs() ** 2).sum().item() / self.n ** 3
        gravity = (self.density() * (0.5 * self.V + self.V_ext)).sum().item()
        return (kinetic + gravity) * self.dx ** 3

    def momentum(self):
        # Total momentum, sum of Im(conj(psi) grad psi) dx^3
        psi_k = torch.fft.fftn(self.psi)
        shapes = [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]
        grad = [torch.fft.ifftn(1j * self.k.reshape(s) * psi_k) for s in shapes]
        return np.array([(self.psi.conj() * g).imag.sum().item() * self.dx ** 3 for g in grad])

# A point mass (black hole) in the box
def separation(box, position):
    # x - position along each axis, nearest periodic image
    x = (torch.arange(box.n, dtype=torch.float64) + 0.5) * box.dx - box.L / 2
    d = [((x - p + box.L / 2) % box.L - box.L / 2).to(box.psi.real.dtype).to(box.psi.device) for p in position]
    return d[0][:, None, None], d[1][None, :, None], d[2][None, None, :]

def point_mass(box, G, M, position, softening):
    # Potential -G M / sqrt(r^2 + eps^2) of a point mass, softened over eps (a grid cannot hold 1/r at r = 0)
    dx, dy, dz = separation(box, position)
    return -G * M / torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2)

def pull(box, G, position, softening):
    # Acceleration of the point mass towards the field: exactly the reaction to point_mass, so momentum is conserved
    dx, dy, dz = separation(box, position)
    weight = box.density() * (dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2) ** -1.5
    return np.array([G * (weight * d).sum().item() * box.dx ** 3 for d in (dx, dy, dz)])

# Initial states for the cosmological box
def coordinates(n, L):
    # Distances along each axis from the centre of the box, for the cell centres
    x = (np.arange(n) + 0.5) * L / n - L / 2
    return x[:, None, None], x[None, :, None], x[None, None, :]

def blobs(n, L, count, mass, width, spread, seed=3):
    # Haloes and raw material from which a soliton forms by itself
    rng = np.random.default_rng(seed)

    X, Y, Z = coordinates(n, L)
    
    amplitude = math.sqrt(mass / (2 * math.pi * width ** 2) ** 1.5)
    
    psi = np.zeros((n, n, n), dtype=complex)
    
    for _ in range(count):
        cx, cy, cz = rng.uniform(-spread, spread, 3)
        r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        psi += amplitude * np.exp(-r2 / (4 * width ** 2)) * np.exp(2j * math.pi * rng.uniform())
    return psi

def ground_state(n, L, G, mass, dtau=5e-3, tolerance=1e-8, max_steps=20000, external=None):
    # The lowest energy state of a given mass, found by evolving in imaginary time; external(box) gives a fixed V_ext
    X, Y, Z = coordinates(n, L)
    
    # Gaussian box
    box = Box(np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * (L / 8) ** 2)) + 0j, L, four_pi_G=4 * math.pi * G)
    if external is not None:
        box.V_ext = external(box)

    mu_old = None
    for step in range(max_steps):
        box.psi = box.psi * math.sqrt(mass / box.mass())
        box.V = box.potential(box.density())
        half = torch.exp(-(box.V + box.V_ext) * dtau / 2)
        box.psi = half * torch.fft.ifftn(torch.fft.fftn(half * box.psi) * torch.exp(-box.k2 * dtau / 2))
        if step % 100 == 0:
            box.psi = box.psi * math.sqrt(mass / box.mass())
            box.V = box.potential(box.density())
            mu = (box.energy() + 0.5 * (box.density() * box.V).sum().item() * box.dx ** 3) / mass
            if mu_old is not None and abs(mu / mu_old - 1) < tolerance:
                return box, True
            mu_old = mu
    return box, False

# Measurements
@functools.lru_cache(maxsize=1)
def soliton_shape(r_end=30.0, r0=1e-6):
    # Radial ground state by shooting: lap psi = 2 (V - mu) psi, lap V = psi^2, psi(0) = 1, V(0) = 0
    from scipy.integrate import solve_ivp

    def rhs(r, y, mu):
        psi, dpsi, V, dV = y
        return [dpsi, 2 * (V - mu) * psi - 2 * dpsi / r, dV, psi ** 2 - 2 * dV / r]

    # Stop at a node (mu too high) or when psi turns back up (mu too low)
    node = lambda r, y, mu: y[0]
    node.terminal = True
    turn = lambda r, y, mu: y[1]
    turn.terminal, turn.direction = True, 1

    def shoot(mu):
        # Series start off r = 0
        y0 = [1 - mu * r0 ** 2 / 3, -2 * mu * r0 / 3, r0 ** 2 / 6, r0 / 3]
        return solve_ivp(rhs, (r0, r_end), y0, args=(mu,), events=(node, turn), rtol=1e-12, atol=1e-14, dense_output=True)

    # Bisect on the eigenvalue mu
    lo, hi = 0.01, 5.0
    for _ in range(60):
        mu = 0.5 * (lo + hi)
        if shoot(mu).t_events[0].size:
            hi = mu
        else:
            lo = mu

    # Nodeless solution, kept up to where it starts to diverge
    sol = shoot(lo)
    r = np.linspace(r0, sol.t[-1], 4000)
    log_f = 2 * np.log(sol.sol(r)[0])

    # Rescale so that rho(0) = 1 and rho(1) = 1/2; r_half fixes rho_c r_c^4 = r_half^4 (hbar/m)^2 / (4 pi G)
    r_half = np.interp(-math.log(2), log_f[::-1], r[::-1])
    return r / r_half, log_f, r_half

def soliton_profile(r, rho_c, r_c):
    # Numerical ground state, rescaled: rho(r) = rho_c f(r / r_c) with f(1) = 1/2
    x, log_f, _ = soliton_shape()
    xr = np.asarray(r) / r_c

    # Exponential tail beyond the tabulated range
    slope = (log_f[-1] - log_f[-2]) / (x[-1] - x[-2])
    return rho_c * np.exp(np.where(xr <= x[-1], np.interp(xr, x, log_f), log_f[-1] + slope * (xr - x[-1])))

def radial_profile(rho, dx, r_max):
    # Mean density in spherical shells of width dx/2 around the densest cell
    n = rho.shape[0]

    # Index of the densest cell
    centre = np.unravel_index(np.argmax(rho), rho.shape)

    # Periodic wrap to [-n/2, n/2)
    d = [(np.arange(n) - c + n // 2) % n - n // 2 for c in centre]
    r = dx * np.sqrt(d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2)

    # Shell index of each cell
    shell = (r / (dx / 2)).astype(int).ravel()

    # Only cells inside r_max
    keep = shell < r_max / (dx / 2)

    # Cells per shell
    count = np.bincount(shell[keep])

    # Skip empty shells
    good = count > 0

    # Mean radius per shell
    r_mean = np.bincount(shell[keep], weights=r.ravel()[keep])[good] / count[good]

    # Mean density per shell
    rho_mean = np.bincount(shell[keep], weights=rho.ravel()[keep])[good] / count[good]
    return r_mean, rho_mean

def fit_core(r, rho):
    # Fit soliton_profile to the inner part (r < 2.5 r_c) of a radial profile
    from scipy.optimize import curve_fit

    # Fit in log space so the tail weighs as much as the peak
    def log_model(r, log_rho_c, r_c):
        return np.log(soliton_profile(r, np.exp(log_rho_c), r_c))

    # First guess: central density and half-density radius
    p = (math.log(rho[0]), r[np.argmax(rho < rho[0] / 2)])

    # Refit as the window r < 2.5 r_c settles
    for _ in range(4):
        inner = r <= 2.5 * p[1]
        p, _ = curve_fit(log_model, r[inner], np.log(rho[inner]), p0=p, bounds=([-np.inf, 0], np.inf))

    return math.exp(p[0]), p[1]

def power_spectrum(rho, L, bins=40):
    # P(k) of the density contrast delta = rho/<rho> - 1 averaged in shells of |k|
    n = rho.shape[0]
    delta_k = np.fft.fftn(rho / rho.mean() - 1.0)

    # Angular wavenumbers per axis
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)

    # |k| per mode
    magk = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2).ravel()

    # Normalised so P has units of volume
    P = np.abs(delta_k.ravel()) ** 2 * L ** 3 / n ** 6

    # From the box mode to Nyquist
    edges = np.linspace(2 * np.pi / L, np.pi * n / L, bins + 1)

    # Bin index of each mode
    which = np.digitize(magk, edges) - 1

    # Drop k = 0 and modes beyond Nyquist
    keep = (which >= 0) & (which < bins)

    # Modes per bin
    count = np.bincount(which[keep], minlength=bins)

    # Shell average
    P_mean = np.bincount(which[keep], weights=P[keep], minlength=bins) / np.maximum(count, 1)

    # Bin centres and P, skipping empty bins
    return 0.5 * (edges[1:] + edges[:-1])[count > 0], P_mean[count > 0]