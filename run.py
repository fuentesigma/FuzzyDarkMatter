#!/usr/bin/env python3
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import os
import sys
import time
import math

import numpy as np
import torch

from universe import Universe, load_power_spectrum, initial_wave, KPC
from wave import Box, blobs, ground_state, radial_profile, fit_core, soliton_profile, power_spectrum, point_mass, pull, separation

HERE = os.path.dirname(os.path.abspath(__file__))

COSMO = dict(
    H0=70.0, Omega_m=0.27, Omega_L=0.73,     # background cosmology (flat)
    m_boson_eV=8.0e-23,                      # boson mass
    L_Mpc=4.0,                               # comoving box side
    n=128,                                   # cells per side
    z_start=1000.0, z_end=10.0,              # redshift range
    pk_file="data/psidm_pk_z1000.txt",       # linear P(k) at z_start: k [1/Mpc], P [Mpc^3]
    seed=1,
    c_kin=0.265, c_pot=1.885,                # time step: dt <= c_kin dx^2 and dt <= c_pot / (a max|V|) (Schive et al. 2014)
    dlna_max=0.1,                            # ... and ln a changes by at most this per step
    snapshot_every=50,                       # steps between saved snapshots
    device="auto", dtype="complex64",        # auto: the GPU (mps or cuda) if there is one
    out="results/cosmo",
)

SOLITON = dict(
    n=128, L=30.0, G=1.0,                    # static box, code units with hbar = m = 1
    blobs=8, blob_mass=0.6, blob_width=1.5, spread=5.0, seed=3,
    t_end=120.0, fits=40,                    # run time, and how many times the core is fitted
    c_kin=0.25, c_pot=0.5,
    ground_n=64, ground_L=10.0, ground_mass=3.0,   # the reference ground state
    device="cpu", dtype="complex128",        # double precision: the mass is conserved to 1e-12
    out="results/soliton",
)

BLACKHOLE = dict(
    n=80, L=10.0, G=1.0,                     # static box, code units with hbar = m = 1
    mass=3.0,                                # soliton mass (core radius ~0.9 for G = 1)
    softening=2.0,                           # black hole potential softened over this many cells (1 conserves momentum poorly)
    ratios="0,0.1,0.25,0.5,1",               # M_bh / M_soliton for the ground states
    tolerance=1e-7,                          # ground state: stop when mu changes less than this per 100 steps
    sink_ratio=0.1,                          # M_bh / M_soliton for the black hole that sinks
    sink_offset=1.0,                         # its starting distance from the centre, released at rest
    t_end=40.0, frames=400,                  # run time of the sinking black hole, and how many records
    c_kin=0.25, c_pot=0.5,
    device="cpu", dtype="complex128",
    out="results/blackhole",
)

# Command line
USAGE = """usage:
    python run.py cosmo        a cosmological box from z_start to z_end   -> results/cosmo/
    python run.py soliton      blobs collapse into a soliton               -> results/soliton/
    python run.py blackhole    a black hole in a soliton                   -> results/blackhole/
    python run.py check        tests of the physics against exact answers

change any setting from the command line, e.g.   python run.py soliton t_end=60 blobs=12"""

def device_of(settings):
    name = settings["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and settings["dtype"] == "complex128":
        sys.exit("the Apple GPU (mps) has no double precision: use dtype=complex64 or device=cpu")
    return torch.device(name)

def outdir(settings):
    path = os.path.join(HERE, settings["out"])
    os.makedirs(path, exist_ok=True)
    return path

# Cosmological box
def cosmo(s):
    out = outdir(s)
    log = open(os.path.join(out, "log.txt"), "w")

    def say(line):
        print(line, flush=True)
        log.write(line + "\n")

    universe = Universe(s["H0"], s["Omega_m"], s["Omega_L"], s["m_boson_eV"])
    a, a_end = 1.0 / (1.0 + s["z_start"]), 1.0 / (1.0 + s["z_end"])
    P_of_k = load_power_spectrum(os.path.join(HERE, s["pk_file"]))
    psi = initial_wave(s["n"], s["L_Mpc"], universe, P_of_k, a, s["seed"])
    device = device_of(s)
    box = Box(torch.tensor(psi, dtype=getattr(torch, s["dtype"]), device=device), universe.code_length(s["L_Mpc"]))

    cell_kpc = s["L_Mpc"] * 1e3 / s["n"]
    say(f"box {s['L_Mpc']} Mpc, {s['n']}^3 cells of {cell_kpc:.1f} kpc (comoving), m = {s['m_boson_eV']:.1e} eV, "
        f"code length {universe.L0 / KPC:.2f} kpc, device {device}")
    save_snapshot(box, a, 0, out, s["L_Mpc"])
    k, P_start = power_spectrum(box.density().cpu().numpy(), s["L_Mpc"])

    step, tau, t0 = 0, 0.0, time.time()
    while a < a_end:
        rate = universe.da_dtau(a)
        dt = min(box.dt_max(a, s["c_kin"], s["c_pot"]), s["dlna_max"] * a / rate)
        if a + rate * dt > a_end:                          # do not overshoot z_end
            dt = (a_end - a) / rate
        a_mid = universe.advance(a, dt / 2)                # the kick uses a at mid-step: second order in dt
        box.step(dt, a_mid)
        a = universe.advance(a, dt)
        tau, step = tau + dt, step + 1
        if step % s["snapshot_every"] == 0 or a >= a_end:
            say(f"step {step:6d}   z = {1 / a - 1:8.3f}   max density {box.density().max().item():9.2f}   "
                f"under-resolved {under_resolved(box):5.1%}   wall {time.time() - t0:7.1f} s")
            save_snapshot(box, a, step, out, s["L_Mpc"])

    k, P_end = power_spectrum(box.density().cpu().numpy(), s["L_Mpc"])
    np.savetxt(os.path.join(out, "power_spectrum.txt"), np.column_stack([k, P_start, P_end]),
               header=f"k [1/Mpc]   P(k) [Mpc^3] at z = {s['z_start']:g}   and at z = {1 / a - 1:g}")
    plot_cosmo(out, k, P_start, P_end, s, a)
    say(f"done: {step} steps; results in {out}")

# Unresolved regions
def under_resolved(box):
    """Fraction of cells where the wave turns by more than pi/2 between neighbours: 
    i.e. the de Broglie wavelength is shorter than 4 cells. 
    
    The phase jump from cell i to i+1 is the angle of psi_{i+1} conj(psi_i)
    .: a jump of pi means 2 cells per wavelength, where the grid gives up.
    """
    worst = torch.zeros_like(box.psi.real)
    for axis in range(3):
        turn = torch.roll(box.psi, -1, axis) * box.psi.conj()
        worst = torch.maximum(worst, torch.atan2(turn.imag, turn.real).abs())
    return (worst > math.pi / 2).float().mean().item()

# Soliton formation
def soliton(s):
    """Blobs with random phases collapse, the core that forms should be the ground state.

    1.- All solitons are one shape rescaled (psi(x) -> lambda^2 psi(lambda x)) 
    2.- rho_c r_c^4 is the same number for every soliton
    3.- I compute it once from the ground state
    4.- Then check that the core that forms by itself has r_c = (that number / rho_c)^(1/4), with no free parameter.
    """
    out = outdir(s)
    device = device_of(s)
    print("reference ground state ...", flush=True)
    ref, converged = ground_state(s["ground_n"], s["ground_L"], s["G"], s["ground_mass"])
    r, rho = radial_profile(ref.density().numpy(), ref.dx, s["ground_L"] / 2)
    rho_c, r_c = fit_core(r, rho)
    invariant = rho_c * r_c ** 4
    print(f"  converged {converged}, r_c = {r_c:.3f} ({r_c / ref.dx:.1f} cells), rho_c r_c^4 = {invariant:.4e}")

    psi = blobs(s["n"], s["L"], s["blobs"], s["blob_mass"], s["blob_width"], s["spread"], s["seed"])
    box = Box(torch.tensor(psi, dtype=getattr(torch, s["dtype"]), device=device), s["L"], 4 * math.pi * s["G"])
    M0, E0 = box.mass(), box.energy()
    t, rows = 0.0, []
    for t_fit in np.linspace(0, s["t_end"], s["fits"] + 1)[1:]:
        while t < t_fit - 1e-12:
            dt = min(box.dt_max(1.0, s["c_kin"], s["c_pot"]), t_fit - t)
            box.step(dt)
            t += dt
        r, rho = radial_profile(box.density().cpu().numpy(), box.dx, s["L"] / 2)
        rho_c, r_c = fit_core(r, rho)
        predicted = (invariant / rho_c) ** 0.25
        rows.append((t, box.mass() / M0 - 1, box.energy() / E0 - 1, rho_c, r_c, predicted))
        print(f"  t = {t:6.1f}   mass drift {rows[-1][1]:+.1e}   energy drift {rows[-1][2]:+.1e}   "
              f"r_c = {r_c:.3f} ({r_c / box.dx:.1f} cells)   predicted {predicted:.3f}   ratio {r_c / predicted:.3f}",
              flush=True)
    rows = np.array(rows)
    np.savetxt(os.path.join(out, "cores.txt"), rows, header="t  mass_drift  energy_drift  rho_c  r_c  r_c_predicted")
    late = rows[rows[:, 0] >= s["t_end"] / 2]
    print(f"late times: r_c / predicted = {np.mean(late[:, 4] / late[:, 5]):.3f} +- {np.std(late[:, 4] / late[:, 5]):.3f}")
    plot_soliton(out, rows, box, r, rho, rho_c, r_c)

# Black hole in a soliton
def blackhole(s):
    """A (Newtonian) point mass in a soliton, softened over a couple of cells.

    1.- Ground states with the black hole at the centre, as M_bh / M_soliton grows: the core contracts. When the black
        hole dominates, the field is the hydrogen-like ground state of -G M_bh / r, with Bohr radius a0 = 1 / (G M_bh)
        (hbar = m = 1) and half its mass inside 1.337 a0.
    2.- A black hole released at rest off-centre: it falls through the soliton, which recoils (the total momentum
        stays zero). Does it sink to the centre, and how fast?
    Physics at the horizon (accretion, superradiance) needs general relativity: out of these equations.
    """
    out = outdir(s)
    device = device_of(s)
    G, M = s["G"], s["mass"]
    eps = s["softening"] * s["L"] / s["n"]
    ratios = [float(q) for q in s["ratios"].split(",")]

    # 1. Ground states with the black hole at the centre
    profiles, rows = [], []
    for q in ratios:
        t0 = time.time()
        box, converged = ground_state(s["n"], s["L"], G, M, tolerance=s["tolerance"],
                                      external=lambda b: point_mass(b, G, q * M, (0, 0, 0), eps))
        rho = box.density().numpy()
        dx, dy, dz = separation(box, (0, 0, 0))
        r = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2).numpy().ravel()
        order = np.argsort(r)
        r_half = r[order][np.searchsorted(np.cumsum(rho.ravel()[order]) * box.dx ** 3, M / 2)]
        rr, prof = radial_profile(rho, box.dx, s["L"] / 2)
        profiles.append((q, rr, prof))
        a0 = 1 / (G * q * M) if q > 0 else math.inf
        rows.append((q, rho.max(), r_half, a0))
        print(f"  M_bh / M = {q:5.2f}: converged {converged}, peak density {rho.max():8.3f}, half-mass radius {r_half:.3f}"
              f"{f', Bohr radius {a0:.3f} ({a0 / box.dx:.1f} cells)' if q > 0 else ''}   [{time.time() - t0:.0f} s]", flush=True)
        if q == 0:
            soliton = box
    rows = np.array(rows)
    np.savetxt(os.path.join(out, "ground_states.txt"), rows, header="M_bh/M  peak_density  half_mass_radius  bohr_radius")

    # 2. A black hole released at rest off-centre, both advanced kick-drift-kick
    box = Box(soliton.psi.to(getattr(torch, s["dtype"])).to(device), s["L"], 4 * math.pi * G)
    M_bh = s["sink_ratio"] * M
    x_bh, v_bh = np.array([s["sink_offset"], 0.0, 0.0]), np.zeros(3)
    box.V_ext = point_mass(box, G, M_bh, x_bh, eps)
    acc = pull(box, G, x_bh, eps)
    energy = lambda: box.energy() + 0.5 * M_bh * v_bh @ v_bh
    E0, P0 = energy(), box.momentum() + M_bh * v_bh
    print(f"sinking: M_bh = {M_bh:.3f} ({s['sink_ratio']} of the soliton), released at rest at x = {s['sink_offset']}", flush=True)

    t, track = 0.0, []
    for t_frame in np.linspace(0, s["t_end"], s["frames"] + 1):
        while t < t_frame - 1e-12:
            dt = min(box.dt_max(1.0, s["c_kin"], s["c_pot"]), t_frame - t)
            v_bh += acc * dt / 2
            x_bh = (x_bh + v_bh * dt + s["L"] / 2) % s["L"] - s["L"] / 2
            box.step(dt, V_ext_end=point_mass(box, G, M_bh, x_bh, eps))
            acc = pull(box, G, x_bh, eps)
            v_bh += acc * dt / 2
            t += dt
        centre = core_centre(box)
        d = (x_bh - centre + s["L"] / 2) % s["L"] - s["L"] / 2
        track.append((t, *x_bh, *centre, np.linalg.norm(d)))
        if (len(track) - 1) % max(s["frames"] // 20, 1) == 0:
            P = box.momentum() + M_bh * v_bh
            print(f"  t = {t:6.2f}   distance to the core {np.linalg.norm(d):.3f}   speed {np.linalg.norm(v_bh):.3f}   "
                  f"energy drift {energy() / E0 - 1:+.1e}   momentum {np.linalg.norm(P - P0):.1e} (of {M_bh * np.linalg.norm(v_bh):.1e})",
                  flush=True)
    track = np.array(track)
    np.savetxt(os.path.join(out, "sinking.txt"), track, header="t  x_bh y_bh z_bh  x_core y_core z_core  distance")
    plot_blackhole(out, s, profiles, rows, track, box, x_bh)
    print(f"figures in {out}")

def core_centre(box, radius=1.0, iterations=20):
    # Centre of the core: the centre of mass of the field under a Gaussian window of this radius, starting from the
    # densest cell and re-centring the window until it settles (a smooth window, since a sharp sphere on the grid can
    # lock off-centre). A mean over the whole periodic box drifts as the tail and the matter thrown out wrap around
    # the edges, even with the momentum conserved
    rho = box.density().cpu().double()
    peak = np.unravel_index(torch.argmax(rho).item(), rho.shape)
    centre = (np.array(peak) + 0.5) * box.dx - box.L / 2
    for _ in range(iterations):
        d = [q.cpu().double() for q in separation(box, centre)]
        weight = rho * torch.exp(-(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) / (2 * radius ** 2))
        offset = np.array([(weight * q).sum().item() for q in d]) / weight.sum().item()
        centre = (centre + offset + box.L / 2) % box.L - box.L / 2
    return centre

def softened_half_radius(GM, softening, N=20000):
    # Half-mass radius of the ground state of -GM / sqrt(r^2 + eps^2) alone (no self-gravity), by a 1D radial solve
    from scipy.linalg import eigh_tridiagonal
    h = (20 / GM + 5 * softening) / N
    r = h * np.arange(1, N + 1)
    u = eigh_tridiagonal(1 / h ** 2 - GM / np.sqrt(r ** 2 + softening ** 2), -0.5 / h ** 2 * np.ones(N - 1),
                         select="i", select_range=(0, 0))[1][:, 0]
    return r[np.searchsorted(np.cumsum(u ** 2) / np.sum(u ** 2), 0.5)]

# Plots and snapshots
def save_snapshot(box, a, step, out, L_Mpc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rho = box.density().cpu().numpy()
    np.savez(os.path.join(out, f"snap_{step:06d}.npz"), psi=box.psi.cpu().numpy(), a=a)
    plt.figure(figsize=(5, 5))
    plt.imshow(np.log10(rho.mean(axis=2)).T, origin="lower", extent=[0, L_Mpc, 0, L_Mpc], cmap="inferno")
    plt.title(f"z = {1 / a - 1:.2f}   (log projected density)")
    plt.xlabel("Mpc"); plt.ylabel("Mpc")
    plt.savefig(os.path.join(out, f"snap_{step:06d}.png"), dpi=100, bbox_inches="tight")
    plt.close()

def plot_cosmo(out, k, P_start, P_end, s, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.loglog(k, P_start, label=f"z = {s['z_start']:g}")
    plt.loglog(k, P_end, label=f"z = {1 / a - 1:g}")
    plt.xlabel("k [1/Mpc]"); plt.ylabel("P(k) [Mpc$^3$]"); plt.legend()
    plt.savefig(os.path.join(out, "power_spectrum.png"), dpi=110, bbox_inches="tight")
    plt.close()

def plot_soliton(out, rows, box, r, rho, rho_c, r_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].imshow(np.log10(box.density().cpu().numpy().sum(axis=2)).T, origin="lower", cmap="inferno")
    ax[0].set_title("projected density at the end")
    ax[1].loglog(r, rho, "o", ms=3, label="measured"); ax[1].loglog(r, soliton_profile(r, rho_c, r_c), label="soliton fit")
    ax[1].set_xlabel("r"); ax[1].set_ylabel("density"); ax[1].legend()
    ax[2].plot(rows[:, 0], rows[:, 4] / rows[:, 5]); ax[2].axhline(1, color="k", lw=0.8)
    ax[2].set_xlabel("t"); ax[2].set_ylabel("r_c / predicted")
    plt.savefig(os.path.join(out, "soliton.png"), dpi=110, bbox_inches="tight")
    plt.close()

def plot_blackhole(out, s, profiles, rows, track, box, x_bh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotstyle
    plotstyle.use()

    # Ground states: profiles (darker = heavier black hole), half-mass radius and peak density
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    shades = plotstyle.ramp(plotstyle.SERIES[0], len(profiles))[::-1]
    for (q, r, rho), color in zip(profiles, shades):
        ax[0].loglog(r, rho, color=color, label=f"{q:g}")
    ax[0].set_xlabel("r"); ax[0].set_ylabel("density"); ax[0].set_ylim(1e-6, None)
    ax[0].legend(title="$M_\\mathrm{bh}/M$", loc="lower left")
    ax[0].set_title("Density profiles")
    q, peak, r_half, a0 = rows.T
    ax[1].plot(q, r_half / r_half[0], "o-", color=plotstyle.SERIES[0], mec="white", mew=1.5, label="measured")
    q_line = np.linspace(q[q > 0].min() / 2, q.max(), 100)
    # The black hole alone, with the same softening as the run; the unsoftened 1.337 a0 is what a finer grid would approach
    eps = s["softening"] * s["L"] / s["n"]
    alone = [softened_half_radius(s["G"] * x * s["mass"], eps) for x in q_line]
    ax[1].plot(q_line, np.array(alone) / r_half[0], color=plotstyle.MUTED, lw=1.2,
               label=f"black hole alone, softened over {eps:g}")
    ax[1].plot(q_line, 1.337 / (s["G"] * q_line * s["mass"]) / r_half[0], color=plotstyle.MUTED, lw=1, ls="--",
               label="black hole alone, unsoftened: $1.337\\,a_0$")
    ax[1].set_ylim(0, 1.1); ax[1].set_xlabel("$M_\\mathrm{bh}/M$"); ax[1].set_ylabel("half-mass radius / without black hole")
    ax[1].legend(loc="upper right"); ax[1].set_title("The core contracts")
    ax[2].plot(q, peak / peak[0], "o-", color=plotstyle.SERIES[0], mec="white", mew=1.5)
    ax[2].set_xlabel("$M_\\mathrm{bh}/M$"); ax[2].set_ylabel("peak density / without black hole"); ax[2].set_title("and densifies")
    plt.savefig(os.path.join(out, "ground_states.png"), dpi=150)
    plt.close()

    # Sinking: distance to the core against time, and the path over the projected density at the end
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax[0].plot(track[:, 0], track[:, 7], color=plotstyle.SERIES[1])
    ax[0].set_xlabel("t"); ax[0].set_ylabel("distance from the black hole to the core"); ax[0].set_ylim(0, None)
    ax[0].set_title(f"A black hole of {s['sink_ratio']:g} soliton masses, released at rest")
    L = s["L"]
    image = ax[1].imshow(np.log10(box.density().cpu().numpy().sum(axis=2) * box.dx).T, origin="lower", cmap="Blues",
                         extent=[-L / 2, L / 2, -L / 2, L / 2])
    image.set_clim(image.get_clim()[1] - 3, image.get_clim()[1])
    fig.colorbar(image, ax=ax[1], shrink=0.8, label="log$_{10}$ projected density").outline.set_visible(False)
    ax[1].plot(track[:, 1], track[:, 2], color=plotstyle.SERIES[1], lw=1, label="black hole")
    ax[1].plot(track[:, 4], track[:, 5], color=plotstyle.INK, lw=1, label="core centre")
    ax[1].plot(*x_bh[:2], "o", color=plotstyle.SERIES[1], mec="white", mew=1.5)
    w = 2.5 * s["sink_offset"]
    ax[1].set_xlim(-w, w); ax[1].set_ylim(-w, w); ax[1].grid(False)
    ax[1].set_xlabel("x"); ax[1].set_ylabel("y"); ax[1].legend(loc="upper right"); ax[1].set_title("Paths, over the density at the end")
    plt.savefig(os.path.join(out, "sinking.png"), dpi=150)
    plt.close()

# Sanity checks
def check():
    results = []

    def report(name, passed, detail):
        results.append(passed)
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)

    # 1. Poisson: for rho = 1 + 0.5 cos(2x), lap V = rho - 1 gives V = -0.5 cos(2x) / 4
    n, L = 32, 2 * math.pi
    x = np.arange(n) * L / n
    box = Box(torch.tensor(np.sqrt(1 + 0.5 * np.cos(2 * x))[:, None, None] * np.ones((1, n, n)) + 0j), L)
    error = np.abs(box.V.numpy()[:, 0, 0] + 0.125 * np.cos(2 * x)).max() / 0.125
    report("Poisson equation, one Fourier mode", error < 1e-10, f"relative error {error:.1e}")

    # 2. A free Gaussian packet spreads as width^2 = 1 + (t/2)^2 (its initial width is 1)
    n, L, t = 64, 20.0, 2.0
    X, Y, Z = (np.arange(n)[:, None, None] * L / n - L / 2, np.arange(n)[None, :, None] * L / n - L / 2,
               np.arange(n)[None, None, :] * L / n - L / 2)
    r2 = X ** 2 + Y ** 2 + Z ** 2
    box = Box(torch.tensor(np.exp(-r2 / 4) + 0j), L, four_pi_G=0.0)
    box.step(t)
    rho = box.density().numpy()
    width = math.sqrt((rho * r2).sum() / rho.sum() / 3)
    report("free particle spreads exactly", abs(width / math.sqrt(1 + (t / 2) ** 2) - 1) < 2e-3,
           f"width {width:.4f}, exact {math.sqrt(1 + (t / 2) ** 2):.4f}")

    # 3. Mass and energy are conserved in a self-gravitating box
    rng = np.random.default_rng(3)
    box = Box(torch.tensor(1 + 0.1 * (rng.normal(size=(32, 32, 32)) + 1j * rng.normal(size=(32, 32, 32)))),
              10.0, four_pi_G=4 * math.pi * 0.05)
    M0, E0 = box.mass(), box.energy()
    for _ in range(400):
        box.step(2e-3)
    dM, dE = abs(box.mass() / M0 - 1), abs(box.energy() / E0 - 1)
    report("mass and energy conserved", dM < 1e-12 and dE < 1e-3, f"mass drift {dM:.0e}, energy drift {dE:.0e} (400 steps)")

    # 4. The time step is second order: halving dt divides the error by 4
    psi0 = torch.tensor(1 + 0.3 * (rng.normal(size=(24, 24, 24)) + 1j * rng.normal(size=(24, 24, 24))))

    def evolve(steps, T=0.2):
        box = Box(psi0.clone(), 10.0)
        for _ in range(steps):
            box.step(T / steps)
        return box.psi

    exact = evolve(2048)
    errors = [(evolve(m) - exact).abs().max().item() for m in (16, 32, 64)]
    order = math.log2(errors[0] / errors[1]) / 2 + math.log2(errors[1] / errors[2]) / 2
    report("kick-drift-kick is second order", 1.8 < order < 2.2, f"measured order {order:.2f}")

    # 5. The ground state does not move under real-time evolution
    ref, converged = ground_state(32, 10.0, 0.5, 50.0)
    rho0 = ref.density().clone()
    for _ in range(200):
        ref.step(1e-4)
    change = ((ref.density() - rho0).abs().max() / rho0.max()).item()
    report("the soliton is stationary", converged and change < 1e-2, f"density changed by {change:.1e} of its peak")

    # 6. Code units: hbar/m = 1 and 4 pi G rho_mean = 1
    from universe import HBAR, G
    u = Universe()
    kinetic, gravity = u.T0 * HBAR / (u.m * u.L0 ** 2), u.T0 ** 2 * 4 * math.pi * G * u.rho_mean
    report("code units", abs(kinetic - 1) < 1e-12 and abs(gravity - 1) < 1e-12, f"hbar/m = {kinetic:.12f}, 4 pi G rho = {gravity:.12f}")

    # 7. A small density wave grows like a (linear theory, matter era) when started in the growing mode
    n, L_Mpc, a = 32, 4.0, 1.0 / 1001.0
    L = u.code_length(L_Mpc)
    kx = 2 * math.pi / L
    x = np.arange(n) * L / n
    delta = 1e-4 * np.cos(kx * x)[:, None, None] * np.ones((1, n, n))
    S = u.growing_mode(a) * delta / kx ** 2                     # lap S = -g delta for a single mode
    box = Box(torch.tensor(np.sqrt(1 + delta) * np.exp(1j * S)), L)
    a0 = a
    while a < 0.02:
        dt = 0.1 * a / u.da_dtau(a)
        box.step(dt, u.advance(a, dt / 2))
        a = u.advance(a, dt)
    grown = np.abs(np.fft.fftn(box.density().numpy() - 1))[1, 0, 0] / np.abs(np.fft.fftn(delta))[1, 0, 0]
    ratio = grown / (a / a0)
    report("linear growth", 0.9 < ratio < 1.1, f"a grew x{a / a0:.1f}; density wave grew x{grown:.1f} (ratio {ratio:.3f})")

    # 8. A point mass alone (no self-gravity): the ground state of -1/sqrt(r^2 + eps^2), against a 1D radial solution
    from scipy.linalg import eigh_tridiagonal
    eps, h, N = 0.3, 1.5e-3, 20000
    r = h * np.arange(1, N + 1)
    exact = eigh_tridiagonal(1 / h ** 2 - 1 / np.sqrt(r ** 2 + eps ** 2), -0.5 / h ** 2 * np.ones(N - 1),
                             select="i", select_range=(0, 0))[0][0]
    box, converged = ground_state(48, 14.0, 0.0, 1.0, tolerance=1e-7, external=lambda b: point_mass(b, 1.0, 1.0, (0, 0, 0), eps))
    mu = box.energy() / box.mass()
    report("ground state of a point mass", converged and abs(mu / exact - 1) < 1e-3, f"energy {mu:.5f}, exact {exact:.5f}")

    # 9. A point mass falling through a soliton: the total momentum (field + point mass) stays zero
    ref, _ = ground_state(32, 10.0, 1.0, 3.0, tolerance=1e-6)
    box = Box(ref.psi, 10.0, 4 * math.pi)
    M_bh, x_bh, v_bh, eps, dt = 0.3, np.array([0.8, 0.3, 0.0]), np.zeros(3), 2 * box.dx, 2e-3
    box.V_ext = point_mass(box, 1.0, M_bh, x_bh, eps)
    acc = pull(box, 1.0, x_bh, eps)
    for _ in range(300):
        v_bh += acc * dt / 2
        x_bh = x_bh + v_bh * dt
        box.step(dt, V_ext_end=point_mass(box, 1.0, M_bh, x_bh, eps))
        acc = pull(box, 1.0, x_bh, eps)
        v_bh += acc * dt / 2
    P = box.momentum() + M_bh * v_bh
    report("momentum conserved with a point mass", np.linalg.norm(P) < 1e-4 * M_bh * np.linalg.norm(v_bh),
           f"total {np.linalg.norm(P):.1e}, point mass alone {M_bh * np.linalg.norm(v_bh):.1e}")

    print(f"\n  {sum(results)} of {len(results)} passed")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("cosmo", "soliton", "blackhole", "check"):
        sys.exit(USAGE)
    command = sys.argv[1]
    settings = {"cosmo": COSMO, "soliton": SOLITON, "blackhole": BLACKHOLE, "check": {}}[command]
    for item in sys.argv[2:]:
        key, value = item.split("=", 1)
        if key not in settings:
            sys.exit(f"unknown setting {key}; the settings are: {', '.join(settings)}")
        settings[key] = type(settings[key])(value) if not isinstance(settings[key], str) else value
    if command == "cosmo":
        cosmo(settings)
    elif command == "soliton":
        soliton(settings)
    elif command == "blackhole":
        blackhole(settings)
    else:
        check()

if __name__ == "__main__":
    main()