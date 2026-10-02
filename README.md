# SP mini

A small Schrödinger–Poisson solver for wave (fuzzy) dark matter, written in Python with PyTorch.
The code is correct and tested. It is kept as a reference. No further development is planned.

## Background

This is a revival of my master's thesis code, written about twelve years ago in C with adaptive mesh refinement (AMR).
The thesis aimed to simulate cosmic structure made of a coherent wave of ultralight bosons. Months before my defence,
Schive, Chiueh & Broadhurst (2014, *Nature Physics* 10, 496) published that result using the GAMER code, and the
project stopped there.

This version rewrites the solver with a modern architecture: a uniform grid, a pseudo-spectral method, and PyTorch so
that it runs on a GPU. **It has no AMR.** That limits it to small problems.

## The equations

A single complex field ψ with density ρ = |ψ|², in units where ħ/m = 1:

    i ∂ψ/∂τ = −½ ∇²ψ + a V ψ,          ∇²V = |ψ|² − 1               (comoving, cosmological box)
    i ∂ψ/∂t = −½ ∇²ψ + (V + V_ext) ψ,  ∇²V = 4πG (ρ − ρ̄)            (static box)

Code units: T₀ = (3/2 H₀² Ω_m)^(−1/2), L₀ = √(T₀ ħ/m), and dτ = dt / (a² T₀).

Each time step is kick–drift–kick (Strang splitting):

- **kick:** the potential rotates the phase at each point, ψ → ψ e^(−i a V Δτ/2);
- **drift:** the kinetic term rotates each Fourier mode,   ψ_k → ψ_k e^(−i k² Δτ/2);
- **Poisson:** solved exactly in Fourier space, V_k = −ρ_k / k².

Each piece is exact, so the mass is conserved to rounding error, and the step is second-order accurate in Δτ.

## Files

| File | Contents |
|---|---|
| `wave.py` | The solver (`Box`): density, Poisson, time step, step limit, mass, energy, momentum. Point mass (softened potential and its back-reaction). Initial blobs. Ground state by imaginary-time evolution. Soliton profile by shooting. Radial profiles, core fit, power spectrum. |
| `universe.py` | Flat ΛCDM background, code units, the expansion a(τ), and the initial wave function from a linear P(k) in the growing mode. |
| `run.py` | All settings and the command line (below). |
| `plotstyle.py` | Shared figure style. |
| `data/psidm_pk_z1000.txt` | Linear power spectrum at z = 1000 (k in 1/Mpc, P in Mpc³), with the wave dark matter cut-off. |
| `data/walker2009_sculptor.fits` | Sculptor stellar velocities (Walker et al. 2009, VizieR J/AJ/137/3100) |

| `results/` | Outputs of the runs below. |

## Usage

Requires Python 3 with `torch`, `numpy`, `scipy` and `matplotlib` (tested with torch 2.6).

    python run.py check        # tests of the physics against exact answers (about 1 min on CPU)
    python run.py cosmo        # cosmological box, z = 1000 -> 10            -> results/cosmo/
    python run.py soliton      # random blobs collapse into a soliton         -> results/soliton/
    python run.py blackhole    # a point mass in a soliton                    -> results/blackhole/

Any setting can be changed on the command line, e.g. `python run.py soliton t_end=60 blobs=12`. The settings and their
defaults are the `COSMO`, `SOLITON` and `BLACKHOLE` dictionaries at the top of `run.py`. `device=auto` uses a GPU (CUDA
or Apple MPS) when one is available. MPS has no double precision, so use `dtype=complex64` there.

## Checks

`python run.py check` runs nine tests. All pass:

1. Poisson equation for one Fourier mode (error about 1e−15).
2. A free Gaussian packet spreads as w² = 1 + (t/2)².
3. Mass and energy are conserved in a self-gravitating box.
4. Kick–drift–kick is second order (measured order 2.01).
5. The soliton ground state stays stationary under real-time evolution.
6. Code units: ħ/m = 1 and 4πG ρ̄ = 1.
7. A small density wave grows like a in the matter era (linear theory).
8. Ground-state energy of a softened point mass, against a 1D radial solution.
9. Total momentum (field plus point mass) is conserved while the point mass falls through a soliton.

The black-hole run was also used as a stress test. Over t = 40, energy stayed constant to about 1e−5 and momentum to
about 1e−6. Halving the time step, or refining the grid from 80³ to 128³, gave the same trajectory.

## Limitations

- **No AMR.** The uniform grid cannot resolve a soliton of about 10 pc inside a box many kpc across. In the default
  cosmological run (4 Mpc, 128³), 68% of cells are under-resolved by z = 10 (fewer than 4 cells per de Broglie
  wavelength). The cosmological results are therefore qualitative only.
- **Point mass:** its potential is softened over 2 cells. The heaviest black-hole ground states (M_bh ≳ 0.5 M_soliton)
  are set mostly by this softening. The figure compares them with a black hole softened the same way.
- **Black-hole experiment:** numerical engineering only. No physical scenario motivates it.