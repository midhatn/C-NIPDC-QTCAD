"""
================================================================================
C-NIPDC-QTCAD Benchmark Framework: Phase 1
Module: m8_verifier.py
Description: 1D Ballistic NEGF-Poisson RTD Solver & Three-Tier Verification Suite
Status: CANDIDATE_PROTOTYPE (UNVERIFIED - Verification Suite Under Diagnostics)
================================================================================
"""

import os
import sys
import json
import time
import platform
import traceback
from datetime import datetime, timezone
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import expit
from scipy.constants import e as q_el, hbar, m_e, epsilon_0, k as k_B

# Physical Constants (GaAs / AlGaAs at 300 K)
EPS_GAAS = 12.9 * epsilon_0
EPS_ALGAAS = 11.9 * epsilon_0
M_GAAS = 0.067 * m_e
M_ALGAAS = 0.092 * m_e
DELTA_EC = 0.23 * q_el          # Conduction band offset (J)
TEMP = 300.0                    # Kelvin
KT = k_B * TEMP                 # Thermal energy (J)
G_SPIN = 2                      # Electron spin degeneracy
G_VALLEY = 1                    # Gamma-valley degeneracy

J_FLOOR = 1e-12                 # Current density numerical floor (A/m^2)
N_FLOOR = 1e15                  # Pointwise carrier density floor (m^-3)
RESIDUAL_ABS_FLOOR = 1e-12      # Absolute denominator floor for Poisson backward error
EDGE_GRID_EPS = 1e-12 * q_el    # Quadrature boundary node offset (J)

# Cross-version NumPy quadrature integration
trapz_integrate = getattr(np, "trapezoid", np.trapz)


# ==============================================================================
# 0. STRICT REPOSITORY GUARDRAIL
# ==============================================================================
def require_snapshot_permission(report):
    """
    Enforces governance gate: blocks formal reference snapshot export unless
    Tier 1, Tier 2, and Tier 3 tests have all executed and passed.
    """
    if (
        report.get("M8_REFERENCE_STATUS") != "VALIDATED_REFERENCE"
        or not report.get("SNAPSHOT_EXPORT_ALLOWED", False)
    ):
        raise RuntimeError(
            "Formal M8 snapshot export is disabled. "
            "Tier 1 (Numerical), Tier 2 (Physical Sanity), and Tier 3 (External Validation) "
            "must all pass first."
        )


# ==============================================================================
# 1. DISCRETE FIELD NORMS
# ==============================================================================
def relative_l2_field_error(field_a, field_b, dx, floor_pointwise=N_FLOOR):
    """Mesh-aware relative L2 norm: ||a - b||_L2 / max(||a||_L2, floor_L2)."""
    l_domain = dx * len(field_a)
    floor_l2 = floor_pointwise * np.sqrt(l_domain)
    
    diff_l2 = np.sqrt(dx * np.sum(np.abs(field_a - field_b)**2))
    norm_a_l2 = np.sqrt(dx * np.sum(np.abs(field_a)**2))
    
    denom = max(norm_a_l2, floor_l2)
    return float(diff_l2 / denom)


# ==============================================================================
# 2. FERMI-DIRAC INVERSION & CONTACT EQUILIBRIUM
# ==============================================================================
def fermi_half(eta):
    """Normalized Fermi-Dirac integral F_{1/2}(eta)."""
    integrand = lambda x: np.sqrt(x) * expit(eta - x)
    val, _ = quad(integrand, 0.0, np.inf, epsabs=1e-11, epsrel=1e-10, limit=200)
    return (2.0 / np.sqrt(np.pi)) * val

def solve_contact_fermi(n_target, m_eff, T=300.0):
    """Calculates Fermi level reproducing contact carrier density."""
    Nc = 2.0 * ((m_eff * k_B * T) / (2.0 * np.pi * hbar**2))**1.5
    res = lambda eta: Nc * fermi_half(eta) - n_target
    eta_sol = brentq(res, -30.0, 30.0, xtol=1e-10)
    return eta_sol * (k_B * T)


# ==============================================================================
# 3. DEVICE GEOMETRY FACTORIES (WITH CELL INVARIANT CHECKS)
# ==============================================================================
class RTDStructure:
    def __init__(self, l_emitter=25e-9, t_barrier=2.6e-9, w_well=5.0e-9, 
                 l_collector=25e-9, n_contact=2e18 * 1e6, n_channel=1e15 * 1e6, 
                 dx=0.2e-9):
        self.dx = dx
        
        self.n_emitter = int(round(l_emitter / dx))
        self.n_b1 = int(round(t_barrier / dx))
        self.n_well = int(round(w_well / dx))
        self.n_b2 = int(round(t_barrier / dx))
        self.n_collector = int(round(l_collector / dx))
        
        self.num_cells = self.n_emitter + self.n_b1 + self.n_well + self.n_b2 + self.n_collector
        self.Nx = self.num_cells + 1
        self.l_tot = self.num_cells * self.dx
        self.x = np.linspace(0.0, self.l_tot, self.Nx)
        
        dx_realized = self.x[1] - self.x[0]
        if not np.isclose(dx_realized, self.dx, rtol=1e-12, atol=1e-15):
            raise ValueError(
                f"Device geometry invariant violation: realized dx ({dx_realized:.6e}) "
                f"does not match specified dx ({self.dx:.6e})."
            )

        self.m_star = np.full(self.Nx, M_GAAS)
        self.eps = np.full(self.Nx, EPS_GAAS)
        self.Ec0 = np.zeros(self.Nx)
        self.Nd = np.full(self.Nx, n_channel)
        
        idx_b1_start = self.n_emitter
        idx_b1_end = idx_b1_start + self.n_b1
        idx_b2_start = idx_b1_end + self.n_well
        idx_b2_end = idx_b2_start + self.n_b2
        
        n_spacer = int(round(5.0e-9 / dx))
        self.Nd[:max(0, self.n_emitter - n_spacer)] = n_contact
        self.Nd[min(self.Nx, idx_b2_end + n_spacer):] = n_contact
        
        for start, end in [(idx_b1_start, idx_b1_end), (idx_b2_start, idx_b2_end)]:
            self.m_star[start:end] = M_ALGAAS
            self.eps[start:end] = EPS_ALGAAS
            self.Ec0[start:end] = DELTA_EC

        self.Ef_contact = solve_contact_fermi(n_contact, M_GAAS, TEMP)

def make_single_barrier_device(l_leads=15e-9, barrier_width=3.0e-9, dx=0.2e-9):
    """Constructs single-barrier benchmark device with integer cell alignment."""
    n_lead = int(round(l_leads / dx))
    n_barr = int(round(barrier_width / dx))
    num_cells = 2 * n_lead + n_barr
    Nx = num_cells + 1
    l_tot = num_cells * dx
    
    dev = RTDStructure(l_emitter=l_leads, t_barrier=barrier_width, w_well=0.0, 
                       l_collector=l_leads, dx=dx)
    dev.Nx = Nx
    dev.l_tot = l_tot
    dev.x = np.linspace(0.0, l_tot, Nx)
    dev.Ec0 = np.zeros(Nx)
    dev.m_star = np.full(Nx, M_GAAS)
    dev.eps = np.full(Nx, EPS_GAAS)
    dev.Nd = np.full(Nx, 1e15 * 1e6)
    
    dev.Ec0[n_lead:n_lead + n_barr] = DELTA_EC
    dev.m_star[n_lead:n_lead + n_barr] = M_ALGAAS
    dev.eps[n_lead:n_lead + n_barr] = EPS_ALGAAS
    return dev


# ==============================================================================
# 4. DISCRETIZATION & ANALYTICAL SURFACE GREEN'S FUNCTIONS
# ==============================================================================
def build_hamiltonian(device, phi):
    """BenDaniel-Duke kinetic operator with face-interpolated 1/m*."""
    Nx, dx = device.Nx, device.dx
    m = device.m_star
    
    inv_m_face = 0.5 * (1.0 / m[:-1] + 1.0 / m[1:])
    t_face = (hbar**2 / (2.0 * dx**2)) * inv_m_face
    
    t_L = hbar**2 / (2.0 * m[0] * dx**2)
    t_R = hbar**2 / (2.0 * m[-1] * dx**2)
    
    d_center = np.zeros(Nx)
    d_center[0] = t_L + t_face[0]
    d_center[-1] = t_face[-1] + t_R
    d_center[1:-1] = t_face[:-1] + t_face[1:]
    
    U = device.Ec0 - q_el * phi
    diag = d_center + U
    H = sp.diags([-t_face, diag, -t_face], [-1, 0, 1], format='csr')
    return H, t_L, t_R, t_face, U[0], U[-1]

def lead_surface_green_function(E, U_lead, t_lead):
    """
    Exact analytical retarded surface Green function for semi-infinite 1D lead.
    Radicands are guarded against negative floating-point artifacts at band edges.
    """
    z = E - (U_lead + 2.0 * t_lead)
    edge = 2.0 * t_lead
    
    if abs(z) < edge:
        root = np.sqrt(max(0.0, edge**2 - z**2))
        return (z - 1j * root) / (2.0 * t_lead**2)
    
    root = np.sqrt(max(0.0, z**2 - edge**2))
    if z <= -edge:
        return (z + root) / (2.0 * t_lead**2)
    else:
        return (z - root) / (2.0 * t_lead**2)


# ==============================================================================
# 5. TRANSPORT EVALUATION & COMPOSITE QUADRATURE
# ==============================================================================
def evaluate_spectral_slice(E, H, t_L, t_R, t_face, U_L, U_R, Nx, dx, device, mu_L, mu_R):
    """Computes spectral densities, transmission, and internal bond currents."""
    g_L = lead_surface_green_function(E, U_L, t_L)
    g_R = lead_surface_green_function(E, U_R, t_R)
    
    sigma_L = t_L**2 * g_L
    sigma_R = t_R**2 * g_R
    
    gamma_L = max(0.0, float(-2.0 * np.imag(sigma_L)))
    gamma_R = max(0.0, float(-2.0 * np.imag(sigma_R)))
    
    A = sp.eye(Nx, format='lil', dtype=complex) * E - H.astype(complex)
    A[0, 0] -= sigma_L
    A[-1, -1] -= sigma_R
    
    solver = spla.factorized(A.tocsc())
    e_1 = np.zeros(Nx); e_1[0] = 1.0
    e_N = np.zeros(Nx); e_N[-1] = 1.0
    
    G_1 = solver(e_1)
    G_N = solver(e_N)
    
    T_E = gamma_L * gamma_R * (np.abs(G_1[-1])**2)
    A_L = gamma_L * (np.abs(G_1)**2)
    A_R = gamma_R * (np.abs(G_N)**2)
    
    c_trans = (G_SPIN * G_VALLEY * device.m_star[0] * KT) / (2.0 * np.pi * hbar**2)
    f_L = c_trans * np.logaddexp(0.0, (mu_L - E) / KT) if E >= U_L else 0.0
    f_R = c_trans * np.logaddexp(0.0, (mu_R - E) / KT) if E >= U_R else 0.0
    
    integrand_n = (1.0 / (2.0 * np.pi * dx)) * (A_L * f_L + A_R * f_R)
    integrand_J_term = (q_el / (2.0 * np.pi * hbar)) * T_E * (f_L - f_R)
    
    w_vec = gamma_L * f_L * G_1[1:] * np.conj(G_1[:-1]) + gamma_R * f_R * G_N[1:] * np.conj(G_N[:-1])
    integrand_J_bond_raw = (q_el / (2.0 * np.pi * hbar)) * (2.0 * t_face * np.imag(w_vec))
    
    return integrand_n, integrand_J_term, integrand_J_bond_raw, T_E

def compute_transport(device, phi, mu_L, mu_R, E_grid):
    """Integrates carrier density, terminal current, and raw bond current."""
    Nx, dx = device.Nx, device.dx
    H, t_L, t_R, t_face, U_L, U_R = build_hamiltonian(device, phi)
    
    n_integrand = np.empty((len(E_grid), Nx))
    J_term_integrand = np.empty(len(E_grid))
    J_bond_integrand = np.empty((len(E_grid), Nx - 1))
    T_spectrum = np.empty(len(E_grid))
    
    for idx, E in enumerate(E_grid):
        dn, dJ_t, dJ_b, T = evaluate_spectral_slice(E, H, t_L, t_R, t_face, U_L, U_R, Nx, dx, device, mu_L, mu_R)
        n_integrand[idx, :] = dn
        J_term_integrand[idx] = dJ_t
        J_bond_integrand[idx, :] = dJ_b
        T_spectrum[idx] = T
        
    n_dens = trapz_integrate(n_integrand, E_grid, axis=0)
    J_terminal = trapz_integrate(J_term_integrand, E_grid)
    J_bond_raw = trapz_integrate(J_bond_integrand, E_grid, axis=0)
    
    return n_dens, J_terminal, J_bond_raw, T_spectrum

def quadratic_segment(energy_start, energy_stop, num_points, reverse=False):
    """Constructs a quadratically clustered sub-segment."""
    if energy_stop <= energy_start or num_points <= 0:
        return np.empty(0)
    u = np.linspace(0.0, 1.0, num_points)
    u_mapped = 1.0 - (1.0 - u)**2 if reverse else u**2
    return energy_start + (energy_stop - energy_start) * u_mapped

def create_adaptive_energy_grid(device, phi, mu_L, mu_R, N_pts=320):
    """
    Composite quadrature grid clustering quadratically around BOTH contact band edges.
    """
    U_L = float(device.Ec0[0] - q_el * phi[0])
    U_R = float(device.Ec0[-1] - q_el * phi[-1])
    
    E_low = min(U_L, U_R)
    E_high = max(U_L, U_R)
    E_max = max(mu_L, mu_R) + 0.30 * q_el
    
    if abs(E_high - E_low) < 1e-6 * q_el:
        u = np.linspace(0.0, 1.0, N_pts)
        return (E_low + EDGE_GRID_EPS) + (E_max - E_low) * (u**2)
        
    edge_window = max(8.0 * KT, 0.06 * q_el)
    
    pts_edge1 = max(40, int(N_pts * 0.25))
    pts_edge2 = max(60, int(N_pts * 0.40))
    pts_tail  = max(30, int(N_pts * 0.20))
    
    parts = []
    E_seg1_end = min(E_high, E_low + edge_window)
    parts.append(quadratic_segment(E_low + EDGE_GRID_EPS, E_seg1_end, pts_edge1, reverse=False))
    
    if E_high > E_seg1_end + 1e-4 * q_el:
        pts_gap = max(20, int(N_pts * 0.15))
        parts.append(np.linspace(E_seg1_end, E_high, pts_gap, endpoint=False))
        
    E_seg2_end = min(E_max, E_high + edge_window)
    parts.append(quadratic_segment(E_high + EDGE_GRID_EPS, E_seg2_end, pts_edge2, reverse=False))
    
    if E_max > E_seg2_end + 1e-4 * q_el:
        parts.append(np.linspace(E_seg2_end, E_max, pts_tail))
        
    grid = np.unique(np.concatenate([p for p in parts if len(p) > 0]))
    return np.sort(grid)


# ==============================================================================
# 6. INTERIOR SPD POISSON & STRICTLY CONVERGED GUMMEL SOLVER
# ==============================================================================
def assemble_interior_poisson(device):
    """Assembles symmetric positive-definite operator L_int on interior nodes."""
    Nx, dx = device.Nx, device.dx
    eps = device.eps
    eps_face = 2.0 / (1.0 / eps[:-1] + 1.0 / eps[1:])
    
    main = (eps_face[:-1] + eps_face[1:]) / dx**2
    off = -eps_face[1:-1] / dx**2
    
    L_int = sp.diags([off, main, off], [-1, 0, 1], format='csc')
    return L_int, eps_face

def solve_poisson(device, L_int, eps_face, n_charge, phi_L, phi_R):
    """Solves L_int * phi_int = rhs_int with Dirichlet potentials in RHS."""
    Nx, dx = device.Nx, device.dx
    rhs = q_el * (device.Nd[1:-1] - n_charge[1:-1])
    
    rhs[0] += eps_face[0] * phi_L / dx**2
    rhs[-1] += eps_face[-1] * phi_R / dx**2
    
    phi = np.empty(Nx)
    phi[0] = phi_L
    phi[-1] = phi_R
    phi[1:-1] = spla.spsolve(L_int, rhs)
    return phi

def evaluate_interior_residual(device, L_int, eps_face, phi, n_calc, phi_L, phi_R):
    """Evaluates backward-error scaled coupled electrostatic residual r_SC."""
    dx = device.dx
    rhs_int = q_el * (device.Nd[1:-1] - n_calc[1:-1])
    rhs_int[0] += eps_face[0] * phi_L / dx**2
    rhs_int[-1] += eps_face[-1] * phi_R / dx**2
    
    L_phi = L_int.dot(phi[1:-1])
    res_int = L_phi - rhs_int
    scale = np.linalg.norm(L_phi) + np.linalg.norm(rhs_int) + RESIDUAL_ABS_FLOOR
    
    r_sc_rel = float(np.linalg.norm(res_int) / scale)
    r_sc_abs = float(np.linalg.norm(res_int))
    r_bc = float(max(abs(phi[0] - phi_L), abs(phi[-1] - phi_R)))
    return r_sc_rel, r_sc_abs, r_bc

def solve_self_consistent_bias(device, V_bias, E_grid=None, phi_init=None, 
                               max_iter=100, tol_sc=1e-4, tol_phi=1e-4, tol_bc=1e-12):
    """
    Self-consistent Gummel solver requiring simultaneous convergence of residual,
    potential updates, and Dirichlet boundaries. Returns comprehensive diagnostics.
    """
    phi_L = 0.0
    phi_R = V_bias
    mu_L = device.Ef_contact
    mu_R = device.Ef_contact - q_el * V_bias
    
    if phi_init is None:
        phi = np.linspace(phi_L, phi_R, device.Nx)
    else:
        phi = phi_init.copy()
        phi[0] = phi_L
        phi[-1] = phi_R

    if E_grid is None:
        E_grid = create_adaptive_energy_grid(device, phi, mu_L, mu_R, N_pts=300)

    L_int, eps_face = assemble_interior_poisson(device)
    
    phi_hist = []
    f_hist = []
    anderson_rejections = 0
    fallback_count = 0
    alpha = 0.10
    prev_r = 1e9
    
    for it in range(max_iter):
        n_calc, _, _, _ = compute_transport(device, phi, mu_L, mu_R, E_grid)
        phi_poiss = solve_poisson(device, L_int, eps_face, n_calc, phi_L, phi_R)
        
        r_sc, r_abs, r_bc = evaluate_interior_residual(device, L_int, eps_face, phi, n_calc, phi_L, phi_R)
        d_phi_inf = float(np.max(np.abs(phi_poiss - phi)))
        
        if r_sc < tol_sc and d_phi_inf < tol_phi and r_bc < tol_bc:
            n_fin, J_fin, J_b_raw, T_fin = compute_transport(device, phi, mu_L, mu_R, E_grid)
            phi_poiss_fin = solve_poisson(device, L_int, eps_face, n_fin, phi_L, phi_R)
            r_fin, r_abs_fin, r_bc_fin = evaluate_interior_residual(device, L_int, eps_face, phi, n_fin, phi_L, phi_R)
            d_phi_fin = float(np.max(np.abs(phi_poiss_fin - phi)))
            converged = (r_fin < tol_sc and d_phi_fin < tol_phi and r_bc_fin < tol_bc)
            
            well_slice = slice(device.n_emitter + device.n_b1, device.n_emitter + device.n_b1 + device.n_well)
            q_well_val = float(trapz_integrate(n_fin[well_slice], device.x[well_slice]))
            
            diag_info = {
                "iterations": it + 1,
                "r_sc_rel": r_fin,
                "r_sc_abs": r_abs_fin,
                "d_phi_inf": d_phi_fin,
                "r_bc": r_bc_fin,
                "anderson_rejections": anderson_rejections,
                "fallbacks": fallback_count,
                "phi_min": float(np.min(phi)),
                "phi_max": float(np.max(phi)),
                "q_well": q_well_val
            }
            return phi, n_fin, J_fin, J_b_raw, r_fin, converged, diag_info
            
        f_curr = phi_poiss - phi
        phi_hist.append(phi.copy())
        f_hist.append(f_curr.copy())
        if len(phi_hist) > 3:
            phi_hist.pop(0)
            f_hist.pop(0)
            
        # Safeguarded Anderson mixing with fallback
        use_anderson = False
        if len(phi_hist) >= 2 and r_sc < prev_r:
            df = f_hist[-1] - f_hist[-2]
            df_norm_sq = float(np.dot(df, df))
            if df_norm_sq > 1e-16:
                gamma = float(np.dot(f_hist[-1], df) / df_norm_sq)
                if abs(gamma) < 0.8:
                    phi_bar = (1.0 - gamma) * phi_hist[-1] + gamma * phi_hist[-2]
                    f_bar = (1.0 - gamma) * f_hist[-1] + gamma * f_hist[-2]
                    beta = min(alpha, 0.10)
                    phi_cand = phi_bar + beta * f_bar
                    if float(np.max(np.abs(phi_cand - phi))) < 0.03:
                        phi_next = phi_cand
                        use_anderson = True

        if not use_anderson:
            if r_sc > prev_r:
                alpha = max(0.02, alpha * 0.5)
                anderson_rejections += 1
                fallback_count += 1
            else:
                alpha = min(0.12, alpha * 1.05)
            phi_next = (1.0 - alpha) * phi + alpha * phi_poiss
            
        prev_r = r_sc
        phi_next[0] = phi_L
        phi_next[-1] = phi_R
        phi = phi_next
        
    n_fin, J_fin, J_b_raw, T_fin = compute_transport(device, phi, mu_L, mu_R, E_grid)
    phi_poiss_fin = solve_poisson(device, L_int, eps_face, n_fin, phi_L, phi_R)
    r_fin, r_abs_fin, r_bc_fin = evaluate_interior_residual(device, L_int, eps_face, phi, n_fin, phi_L, phi_R)
    d_phi_fin = float(np.max(np.abs(phi_poiss_fin - phi)))
    well_slice = slice(device.n_emitter + device.n_b1, device.n_emitter + device.n_b1 + device.n_well)
    q_well_val = float(trapz_integrate(n_fin[well_slice], device.x[well_slice]))
    
    diag_info = {
        "iterations": max_iter,
        "r_sc_rel": r_fin,
        "r_sc_abs": r_abs_fin,
        "d_phi_inf": d_phi_fin,
        "r_bc": r_bc_fin,
        "anderson_rejections": anderson_rejections,
        "fallbacks": fallback_count,
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "q_well": q_well_val
    }
    return phi, n_fin, J_fin, J_b_raw, r_fin, False, diag_info

def solve_with_continuation_bisection(device, v_target, v_start, phi_start, 
                                      E_grid=None, depth=0, max_depth=3, 
                                      tol_sc=1e-4, tol_phi=1e-4):
    """Solves at v_target starting from (v_start, phi_start) with recursive bisection on failure."""
    if phi_start is not None:
        phi_init = phi_start + (v_target - v_start) * (device.x / device.l_tot)
    else:
        phi_init = None
        
    phi, n, J, J_b, r, conv, diag = solve_self_consistent_bias(
        device, v_target, E_grid=E_grid, phi_init=phi_init, 
        max_iter=100, tol_sc=tol_sc, tol_phi=tol_phi
    )
    if conv:
        return phi, n, J, J_b, r, True, diag
        
    if depth < max_depth:
        v_mid = 0.5 * (v_start + v_target)
        phi_m, _, _, _, _, c_m, _ = solve_with_continuation_bisection(
            device, v_mid, v_start, phi_start, E_grid=None, 
            depth=depth + 1, max_depth=max_depth, tol_sc=tol_sc, tol_phi=tol_phi
        )
        if c_m:
            return solve_with_continuation_bisection(
                device, v_target, v_mid, phi_m, E_grid=None, 
                depth=depth + 1, max_depth=max_depth, tol_sc=tol_sc, tol_phi=tol_phi
            )
            
    return phi, n, J, J_b, r, False, diag


# ==============================================================================
# 7. COMPLETE THREE-TIER VERIFICATION HARNESS & GOVERNANCE
# ==============================================================================
class M8VerificationRunner:
    def __init__(self):
        dev = RTDStructure()
        self.report = {
            "M8_REFERENCE_STATUS": "UNVERIFIED",
            "BASELINE_DEVELOPMENT_ALLOWED": False,
            "SNAPSHOT_EXPORT_ALLOWED": False,
            "tier_1_mandatory_passed": False,
            "tier_2_sanity_passed": False,
            "tier_3_raw_test_passed": False,
            "tier_3_validation_passed": False,
            "executed_at_utc": None,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform()
            },
            "parameters": {
                "dx": dev.dx,
                "Nx": dev.Nx,
                "temperature": TEMP,
                "n_contact": 2e18 * 1e6,
                "tol_sc": 1e-4,
                "max_iter": 100
            },
            "summary": {},
            "tests": {}
        }

    def register_test(self, test_id, name, tier, status, metric=None, threshold=None, duration=0.0, details=""):
        self.report["tests"][test_id] = {
            "name": name,
            "tier": tier,
            "status": status,
            "metric": float(metric) if metric is not None else None,
            "threshold": float(threshold) if threshold is not None else None,
            "duration_sec": float(duration),
            "details": details
        }
        metric_str = f"{metric:.3e}" if metric is not None else "N/A"
        thresh_str = f"{threshold:.3e}" if threshold is not None else "N/A"
        print(f"[{status:<15}] {test_id}: {name:<40} Metric: {metric_str} (Tol: {thresh_str}) [{duration:.2f}s]")

    def execute_test(self, test_id, name, tier, test_func):
        t_start = time.perf_counter()
        try:
            passed, metric, threshold, details = test_func()
            duration = time.perf_counter() - t_start
            status = "PASS" if passed else "FAIL"
            self.register_test(test_id, name, tier, status, metric, threshold, duration, details)
        except Exception as exc:
            duration = time.perf_counter() - t_start
            err_details = f"{repr(exc)}\n{traceback.format_exc()}"
            self.register_test(test_id, name, tier, "ERROR", None, None, duration, err_details)

    def run_suite(self):
        print("=" * 80)
        print("Executing Complete M8 Three-Tier Verification Suite")
        print("=" * 80)
        dev = RTDStructure()

        # ----------------------------------------------------------------------
        # Test 01: Fermi-Dirac Inversion Sanity
        # ----------------------------------------------------------------------
        def test_01():
            Nc = 2.0 * ((M_GAAS * KT) / (2.0 * np.pi * hbar**2))**1.5
            eta_val = dev.Ef_contact / KT
            n_reconstructed = Nc * fermi_half(eta_val)
            err = abs(n_reconstructed - dev.Nd[0]) / dev.Nd[0]
            return err < 1e-9, err, 1e-9, f"Target: {dev.Nd[0]:.2e}, Recon: {n_reconstructed:.2e}"
        self.execute_test("Test_01", "Fermi Inversion Sanity", "Tier_1_Core", test_01)

        # ----------------------------------------------------------------------
        # Test 02: Surface Green's Function Branch Selection
        # ----------------------------------------------------------------------
        def test_02():
            H, t_L, t_R, t_f, U_L, U_R = build_hamiltonian(dev, np.zeros(dev.Nx))
            passband = 4.0 * t_L
            E_sub = U_L - 0.1 * q_el
            E_pass = U_L + 0.5 * passband
            g_sub = lead_surface_green_function(E_sub, U_L, t_L)
            g_pass = lead_surface_green_function(E_pass, U_L, t_L)
            decay_ok = abs(t_L * g_sub) <= 1.0 + 1e-10
            imag_ok = np.imag(g_pass) < -1e-15
            passed = decay_ok and imag_ok
            metric = 0.0 if passed else 1.0
            return passed, metric, 0.5, f"Evanescent decay: {decay_ok}, Passband Im: {imag_ok}"
        self.execute_test("Test_02", "Surface GF Branch Selection", "Tier_1_Core", test_02)

        # ----------------------------------------------------------------------
        # Test 03: Uniform-Chain Passband Transmission
        # ----------------------------------------------------------------------
        def test_03():
            dev_flat = RTDStructure()
            dev_flat.Ec0[:] = 0.0
            dev_flat.m_star[:] = M_GAAS
            H_f, t_L, t_R, t_face, U_L, U_R = build_hamiltonian(dev_flat, np.zeros(dev_flat.Nx))
            passband = 4.0 * t_L
            E_test = np.linspace(U_L + 0.05 * passband, U_L + 0.95 * passband, 40)
            errs = [abs(evaluate_spectral_slice(e, H_f, t_L, t_R, t_face, U_L, U_R, dev_flat.Nx, dev_flat.dx, dev_flat, 0.0, 0.0)[3] - 1.0) for e in E_test]
            max_err = float(np.max(errs))
            return max_err < 1e-4, max_err, 1e-4, "Evaluated across [0.05, 0.95] of 4t passband."
        self.execute_test("Test_03", "Uniform Passband Transmission", "Tier_1_Core", test_03)

        # ----------------------------------------------------------------------
        # Test 04: Exact Gauge Invariance under Potential Shift
        # ----------------------------------------------------------------------
        def test_04():
            phi_0 = np.linspace(0.0, 0.08, dev.Nx)
            mu_L0, mu_R0 = dev.Ef_contact, dev.Ef_contact - q_el * 0.08
            E_base = create_adaptive_energy_grid(dev, phi_0, mu_L0, mu_R0, N_pts=220)
            n0, J0, _, _ = compute_transport(dev, phi_0, mu_L0, mu_R0, E_base)
            Delta_V = 0.5
            Delta_E = -q_el * Delta_V
            n_sh, J_sh, _, _ = compute_transport(dev, phi_0 + Delta_V, mu_L0 + Delta_E, mu_R0 + Delta_E, E_base + Delta_E)
            
            rel_n = relative_l2_field_error(n0, n_sh, dev.dx, floor_pointwise=N_FLOOR)
            rel_J = abs(J0 - J_sh) / max(abs(J0), J_FLOOR)
            total_err = float(rel_n + rel_J)
            return total_err < 1e-8, total_err, 1e-8, f"Shift: {Delta_V} V, rel_n: {rel_n:.2e}, rel_J: {rel_J:.2e}"
        self.execute_test("Test_04", "Exact Gauge Invariance", "Tier_1_Core", test_04)

        # ----------------------------------------------------------------------
        # Test 05: Bond-Current vs. Landauer Consistency & Uniformity
        # ----------------------------------------------------------------------
        def test_05():
            phi_test = np.linspace(0.0, 0.15, dev.Nx)
            mu_L = dev.Ef_contact
            mu_R = dev.Ef_contact - q_el * 0.15
            E_grid = create_adaptive_energy_grid(dev, phi_test, mu_L, mu_R, N_pts=240)
            _, J_terminal, J_bond_raw, _ = compute_transport(dev, phi_test, mu_L, mu_R, E_grid)
            
            denom = max(abs(J_terminal), J_FLOOR)
            err_pos = float(np.max(np.abs(J_bond_raw - J_terminal)) / denom)
            err_neg = float(np.max(np.abs(-J_bond_raw - J_terminal)) / denom)
            
            orientation = "+1" if err_pos <= err_neg else "-1"
            orientation_sign = 1.0 if err_pos <= err_neg else -1.0
            oriented_bond_current = orientation_sign * J_bond_raw
            
            matched_err = float(np.max(np.abs(oriented_bond_current - J_terminal)) / denom)
            mean_bond = np.mean(oriented_bond_current)
            uniformity = float(np.max(np.abs(oriented_bond_current - mean_bond)) / max(abs(mean_bond), J_FLOOR))
            
            metric = max(matched_err, uniformity)
            passed = (matched_err < 1e-4) and (uniformity < 1e-4)
            details = (
                f"Inferred orientation: {orientation}; "
                f"Landauer match error: {matched_err:.3e}; "
                f"bond-current uniformity: {uniformity:.3e}"
            )
            return passed, metric, 1e-4, details
        self.execute_test("Test_05", "Bond-Current vs. Landauer Consistency", "Tier_1_Core", test_05)

        # ----------------------------------------------------------------------
        # Test 06: Self-Consistent Zero-Bias Contact Equilibrium
        # ----------------------------------------------------------------------
        def test_06():
            u = np.linspace(0.0, 1.0, 300)
            E_eq = (dev.Ef_contact + 0.35 * q_el) * (u**2)
            phi_eq, n_eq, J_eq, _, r_eq, c_eq, _ = solve_self_consistent_bias(dev, V_bias=0.0, E_grid=E_eq, tol_sc=1e-4)
            v_th = np.sqrt(2.0 * KT / (np.pi * M_GAAS))
            norm_J0 = float(abs(J_eq) / (q_el * dev.Nd[0] * v_th))
            contact_err = float(np.max(np.abs(n_eq[2:12] - dev.Nd[2:12]) / dev.Nd[2:12]))
            metric = max(norm_J0, contact_err)
            passed = c_eq and (norm_J0 < 1e-6) and (contact_err < 2e-2)
            return passed, metric, 2e-2, f"Norm J: {norm_J0:.2e}, Contact Err: {contact_err:.2e}, Conv: {c_eq}"
        self.execute_test("Test_06", "Preliminary Zero-Bias Equilibrium", "Tier_1_Core", test_06)

        # ----------------------------------------------------------------------
        # Test 07: Variable-Mass Barrier Benchmark (Analytic BenDaniel-Duke)
        # ----------------------------------------------------------------------
        def test_07():
            d_b = 3.0e-9
            dev_sb = make_single_barrier_device(l_leads=15e-9, barrier_width=d_b, dx=0.2e-9)
            H_sb, t_Lsb, t_Rsb, t_fsb, U_Lsb, U_Rsb = build_hamiltonian(dev_sb, np.zeros(dev_sb.Nx))
            E_eval = 0.15 * q_el
            _, _, _, T_num = evaluate_spectral_slice(E_eval, H_sb, t_Lsb, t_Rsb, t_fsb, U_Lsb, U_Rsb, dev_sb.Nx, dev_sb.dx, dev_sb, 0.0, 0.0)
            
            k1 = np.sqrt(2.0 * M_GAAS * E_eval) / hbar
            kappa2 = np.sqrt(2.0 * M_ALGAAS * (DELTA_EC - E_eval)) / hbar
            beta1 = k1 / M_GAAS
            beta2 = kappa2 / M_ALGAAS
            T_analytic = 1.0 / (1.0 + 0.25 * ((beta2 / beta1 + beta1 / beta2)**2) * (np.sinh(kappa2 * d_b)**2))
            rel_err = float(abs(T_num - T_analytic) / T_analytic)
            return rel_err < 0.05, rel_err, 0.05, f"T_num: {T_num:.4e}, T_analytic: {T_analytic:.4e}"
        self.execute_test("Test_07", "Variable-Mass Barrier Benchmark", "Tier_1_Mandatory", test_07)

        # ----------------------------------------------------------------------
        # Test 08a: Prescribed-Potential Mesh Check (dx=0.2nm vs dx=0.1nm)
        # ----------------------------------------------------------------------
        def test_08a():
            dev_half = RTDStructure(dx=0.1e-9)
            phi_c = np.linspace(0.0, 0.15, dev.Nx)
            phi_f = np.linspace(0.0, 0.15, dev_half.Nx)
            E_base = create_adaptive_energy_grid(dev, phi_c, dev.Ef_contact, dev.Ef_contact - q_el * 0.15, N_pts=280)
            _, J_c, _, _ = compute_transport(dev, phi_c, dev.Ef_contact, dev.Ef_contact - q_el * 0.15, E_base)
            _, J_f, _, _ = compute_transport(dev_half, phi_f, dev_half.Ef_contact, dev_half.Ef_contact - q_el * 0.15, E_base)
            err = float(abs(J_c - J_f) / max(abs(J_f), J_FLOOR))
            return err < 0.03, err, 0.03, f"dx=0.2nm vs 0.1nm. J_c: {J_c:.3e}, J_f: {J_f:.3e}"
        self.execute_test("Test_08a", "Prescribed-Potential Mesh Check", "Tier_1_Mandatory", test_08a)

        # ----------------------------------------------------------------------
        # Test 08b: Self-Consistent I-V Mesh Convergence (dx vs dx/2)
        # ----------------------------------------------------------------------
        def test_08b():
            V_bias_test = 0.08
            phi_c, _, J_c, _, _, conv_c, _ = solve_self_consistent_bias(dev, V_bias=V_bias_test, tol_sc=1e-4)
            
            dev_half = RTDStructure(dx=0.1e-9)
            phi_init_f = np.interp(dev_half.x, dev.x, phi_c)
            phi_f, _, J_f, _, _, conv_f, _ = solve_self_consistent_bias(dev_half, V_bias=V_bias_test, phi_init=phi_init_f, tol_sc=1e-4)
            
            err_J = float(abs(J_c - J_f) / max(abs(J_f), J_FLOOR))
            phi_f_on_c = np.interp(dev.x, dev_half.x, phi_f)
            err_phi = float(np.linalg.norm(phi_c - phi_f_on_c) / max(np.linalg.norm(phi_c), 1e-4))
            metric = max(err_J, err_phi)
            passed = conv_c and conv_f and (metric < 0.05)
            details = f"V={V_bias_test}V SC mesh. conv_c: {conv_c}, conv_f: {conv_f}, err_J: {err_J:.3e}, err_phi: {err_phi:.3e}"
            return passed, metric, 0.05, details
        self.execute_test("Test_08b", "Self-Consistent I-V Mesh Convergence", "Tier_1_Mandatory", test_08b)

        # ----------------------------------------------------------------------
        # Test 09: Hierarchical Dual Energy Convergence (400, 800, 1600 pts)
        # ----------------------------------------------------------------------
        def test_09():
            phi_c = np.linspace(0.0, 0.10, dev.Nx)
            mu_L = dev.Ef_contact
            mu_R = dev.Ef_contact - q_el * 0.10
            
            E_400 = create_adaptive_energy_grid(dev, phi_c, mu_L, mu_R, N_pts=400)
            E_800 = create_adaptive_energy_grid(dev, phi_c, mu_L, mu_R, N_pts=800)
            E_1600 = create_adaptive_energy_grid(dev, phi_c, mu_L, mu_R, N_pts=1600)
            
            n_400, J_400, _, _ = compute_transport(dev, phi_c, mu_L, mu_R, E_400)
            n_800, J_800, _, _ = compute_transport(dev, phi_c, mu_L, mu_R, E_800)
            n_1600, J_1600, _, _ = compute_transport(dev, phi_c, mu_L, mu_R, E_1600)
            
            err_J_400_800 = float(abs(J_400 - J_800) / max(abs(J_800), J_FLOOR))
            err_n_400_800 = float(relative_l2_field_error(n_400, n_800, dev.dx, floor_pointwise=N_FLOOR))
            
            err_J_800_1600 = float(abs(J_800 - J_1600) / max(abs(J_1600), J_FLOOR))
            err_n_800_1600 = float(relative_l2_field_error(n_800, n_1600, dev.dx, floor_pointwise=N_FLOOR))
            
            well_slice = slice(dev.n_emitter + dev.n_b1, dev.n_emitter + dev.n_b1 + dev.n_well)
            q_well_800 = trapz_integrate(n_800[well_slice], dev.x[well_slice])
            q_well_1600 = trapz_integrate(n_1600[well_slice], dev.x[well_slice])
            err_q_well = float(abs(q_well_800 - q_well_1600) / max(abs(q_well_1600), 1.0))
            
            metric = float(max(err_J_800_1600, err_n_800_1600))
            passed = (err_J_800_1600 < 5e-3) and (err_n_800_1600 < 5e-3)
            details = (
                f"400v800: err_J={err_J_400_800:.2e}, err_n={err_n_400_800:.2e}; "
                f"800v1600: err_J={err_J_800_1600:.2e}, err_n={err_n_800_1600:.2e}, err_q_well={err_q_well:.2e}"
            )
            return passed, metric, 5e-3, details
        self.execute_test("Test_09", "Dual Charge/Current Energy Convergence", "Tier_1_Mandatory", test_09)

        # ----------------------------------------------------------------------
        # Test 10: Coupled Interior Residual Sweep
        # ----------------------------------------------------------------------
        def test_10():
            biases = [0.04, 0.08, 0.12, 0.16]
            max_r = 0.0
            worst_bias = None
            phi_curr = None
            all_conv = True
            
            for vb in biases:
                phi_curr, _, _, _, r_fin, conv, _ = solve_self_consistent_bias(
                    dev, vb, phi_init=phi_curr, tol_sc=5e-5, tol_phi=5e-5
                )
                if r_fin > max_r:
                    max_r = r_fin
                    worst_bias = vb
                if not conv:
                    all_conv = False
                    
            passed = all_conv and (max_r < 1e-4)
            details = f"Biases: {biases}, all_conv: {all_conv}, max_r_SC: {max_r:.3e} (at {worst_bias:.2f}V)"
            return passed, max_r, 1e-4, details
        self.execute_test("Test_10", "Coupled Interior Residual Sweep", "Tier_1_Mandatory", test_10)

        # ----------------------------------------------------------------------
        # Test 11: Forward/Reverse Continuation Consistency (10 mV Grid)
        # ----------------------------------------------------------------------
        def test_11():
            common_biases = np.round(np.arange(0.00, 0.121, 0.01), 4)
            fwd_results = []
            phi_curr = None
            all_fwd_conv = True
            
            for idx, vb in enumerate(common_biases):
                mu_L = dev.Ef_contact
                mu_R = dev.Ef_contact - q_el * vb
                E_grid_vb = create_adaptive_energy_grid(dev, np.linspace(0.0, vb, dev.Nx), mu_L, mu_R, N_pts=320)
                
                v_prev = common_biases[idx - 1] if idx > 0 else 0.0
                phi_curr, n_curr, J_val, J_b_curr, r_curr, conv, diag = solve_with_continuation_bisection(
                    dev, vb, v_prev, phi_curr, E_grid=E_grid_vb, tol_sc=3e-5, tol_phi=3e-5
                )
                if not conv:
                    all_fwd_conv = False
                fwd_results.append({"V": vb, "J": J_val, "phi": phi_curr.copy(), "diag": diag})
                
            rev_results = {}
            all_rev_conv = True
            rev_results[len(common_biases) - 1] = fwd_results[-1]
            phi_curr = fwd_results[-1]["phi"].copy()
            
            for idx in range(len(common_biases) - 2, -1, -1):
                vb = common_biases[idx]
                mu_L = dev.Ef_contact
                mu_R = dev.Ef_contact - q_el * vb
                E_grid_vb = create_adaptive_energy_grid(dev, np.linspace(0.0, vb, dev.Nx), mu_L, mu_R, N_pts=320)
                
                v_next = common_biases[idx + 1]
                phi_curr, n_curr, J_val, J_b_curr, r_curr, conv, diag = solve_with_continuation_bisection(
                    dev, vb, v_next, phi_curr, E_grid=E_grid_vb, tol_sc=3e-5, tol_phi=3e-5
                )
                if not conv:
                    all_rev_conv = False
                rev_results[idx] = {"V": vb, "J": J_val, "phi": phi_curr.copy(), "diag": diag}
                
            all_conv = all_fwd_conv and all_rev_conv
            max_err_J = 0.0
            max_err_phi = 0.0
            worst_j_bias = None
            worst_phi_bias = None
            
            for k, vb in enumerate(common_biases):
                j_f = fwd_results[k]["J"]
                j_r = rev_results[k]["J"]
                err_J_k = float(abs(j_f - j_r) / max(abs(j_f), J_FLOOR))
                err_phi_k = float(relative_l2_field_error(fwd_results[k]["phi"], rev_results[k]["phi"], dev.dx, floor_pointwise=1e-4))
                
                if err_J_k > max_err_J:
                    max_err_J = err_J_k
                    worst_j_bias = vb
                if err_phi_k > max_err_phi:
                    max_err_phi = err_phi_k
                    worst_phi_bias = vb
                
            metric = max(max_err_J, max_err_phi)
            passed = all_conv and (metric < 1e-3)
            
            idx_j_worst = int(np.where(common_biases == worst_j_bias)[0][0])
            details = (
                f"Biases [0-0.12V, 10mV]. all_conv: {all_conv}, "
                f"max_err_J: {max_err_J:.3e} (at {worst_j_bias:.2f}V), "
                f"max_err_phi: {max_err_phi:.3e} (at {worst_phi_bias:.2f}V); "
                f"At worst J ({worst_j_bias:.2f}V): "
                f"J_fwd={fwd_results[idx_j_worst]['J']:.3e} (iters={fwd_results[idx_j_worst]['diag']['iterations']}), "
                f"J_rev={rev_results[idx_j_worst]['J']:.3e} (iters={rev_results[idx_j_worst]['diag']['iterations']})"
            )
            return passed, metric, 1e-3, details
        self.execute_test("Test_11", "Forward/Reverse Continuation Consistency", "Tier_1_Mandatory", test_11)

        # Shared IV Cache
        iv_cache = {}

        # ----------------------------------------------------------------------
        # Test 12: Qualitative NDR Detection (Tier 2 Sanity)
        # ----------------------------------------------------------------------
        def test_12():
            sweep_voltages = np.round(np.arange(0.00, 0.441, 0.02), 4)
            currents = []
            phi_curr = None
            v_prev = 0.0
            
            for idx, v in enumerate(sweep_voltages):
                if v == 0.0:
                    currents.append(0.0)
                    phi_curr = np.zeros(dev.Nx)
                    continue
                phi_curr, _, J_val, _, _, conv, _ = solve_with_continuation_bisection(
                    dev, v, v_prev, phi_curr, max_depth=3, tol_sc=1e-4, tol_phi=1e-4
                )
                if not conv:
                    return False, 0.0, 1.20, f"Bias sweep failed to converge at {v:.3f} V."
                currents.append(J_val)
                v_prev = v
                
            currents = np.asarray(currents, dtype=float)
            voltages = np.asarray(sweep_voltages, dtype=float)
            
            idx_peak_coarse = int(np.argmax(currents))
            v_peak_coarse = voltages[idx_peak_coarse]
            
            idx_valley_cand = idx_peak_coarse + 1 + int(np.argmin(currents[idx_peak_coarse + 1:]))
            v_valley_coarse = voltages[idx_valley_cand]
            
            # Local refinement (5 mV spacing) across the candidate NDR window
            v_refine_start = max(0.02, v_peak_coarse - 0.04)
            v_refine_end = min(0.44, v_valley_coarse + 0.04)
            fine_voltages = np.round(np.arange(v_refine_start, v_refine_end + 0.001, 0.005), 4)
            
            all_voltages = np.unique(np.concatenate([voltages, fine_voltages]))
            all_currents_dict = {v_i: j_i for v_i, j_i in zip(voltages, currents)}
            
            phi_track = None
            v_track_prev = 0.0
            for v in all_voltages:
                if v in all_currents_dict:
                    v_track_prev = v
                    continue
                phi_track, _, J_val, _, _, conv, _ = solve_with_continuation_bisection(
                    dev, v, v_track_prev, phi_track, max_depth=3, tol_sc=1e-4, tol_phi=1e-4
                )
                if not conv:
                    return False, 0.0, 1.20, f"Refined bias sweep failed to converge at {v:.3f} V."
                all_currents_dict[v] = J_val
                v_track_prev = v
                
            sorted_voltages = np.sort(list(all_currents_dict.keys()))
            sorted_currents = np.array([all_currents_dict[v] for v in sorted_voltages])
            
            iv_cache["voltages"] = sorted_voltages.tolist()
            iv_cache["currents"] = sorted_currents.tolist()
            
            idx_peak = int(np.argmax(sorted_currents))
            interior_peak = (0 < idx_peak < len(sorted_currents) - 2)
            
            idx_valley = idx_peak + 1 + int(np.argmin(sorted_currents[idx_peak + 1:]))
            valid_valley = (idx_valley > idx_peak and idx_valley < len(sorted_currents))
            
            slopes = np.diff(sorted_currents) / np.diff(sorted_voltages)
            negative_intervals = int(np.count_nonzero(slopes[idx_peak:] < 0.0))
            
            v_peak = float(sorted_voltages[idx_peak])
            j_peak = float(sorted_currents[idx_peak])
            v_valley = float(sorted_voltages[idx_valley])
            j_valley = float(sorted_currents[idx_valley])
            pvcr = float(j_peak / max(j_valley, J_FLOOR))
            
            iv_cache["v_peak"] = v_peak
            iv_cache["j_peak"] = j_peak
            iv_cache["v_valley"] = v_valley
            iv_cache["j_valley"] = j_valley
            iv_cache["pvcr"] = pvcr
            
            passed = interior_peak and valid_valley and (negative_intervals >= 2) and (pvcr >= 1.20)
            metric = pvcr
            details = (
                f"V_peak: {v_peak:.3f}V, J_peak: {j_peak:.2e} A/m^2, "
                f"V_valley: {v_valley:.3f}V, J_valley: {j_valley:.2e} A/m^2, "
                f"PVCR: {pvcr:.2f}, neg_intervals: {negative_intervals}, "
                f"interior_peak: {interior_peak}, valid_valley: {valid_valley}"
            )
            return passed, metric, 1.20, details
        self.execute_test("Test_12", "Qualitative NDR Detection", "Tier_2_Sanity", test_12)

        # ----------------------------------------------------------------------
        # Test 13: Matched Literature RTD Benchmark (Tier 3 Validation)
        # ----------------------------------------------------------------------
        def test_13():
            tier1_all_ids = [
                tid for tid, tdata in self.report["tests"].items()
                if tdata["tier"].startswith("Tier_1") and tid != "Test_13"
            ]
            t1_passed = len(tier1_all_ids) > 0 and all(
                self.report["tests"][tid]["status"] == "PASS" for tid in tier1_all_ids
            )
            t2_passed = (self.report["tests"].get("Test_12", {}).get("status") == "PASS")
            
            if not t1_passed or not t2_passed:
                self.register_test(
                    "Test_13", "Matched Literature RTD Benchmark", "Tier_3_Validation",
                    "BLOCKED", None, None, 0.0,
                    "Tier 3 validation is strictly blocked until all Tier 1 and Tier 2 tests pass."
                )
                return False, None, None, "BLOCKED: Requires Tier 1 and 2."
                
            self.register_test(
                "Test_13", "Matched Literature RTD Benchmark", "Tier_3_Validation",
                "NOT_IMPLEMENTED", None, None, 0.0,
                "Requires external citation-matched benchmark structure implementation."
            )
            return False, None, None, "NOT_IMPLEMENTED"
            
        test_13()

        # ----------------------------------------------------------------------
        # THREE-TIER PROMOTION GOVERNANCE LOGIC
        # ----------------------------------------------------------------------
        core_ids = ["Test_01", "Test_02", "Test_03", "Test_04", "Test_05", "Test_06"]
        core_passed = all(self.report["tests"][tid]["status"] == "PASS" for tid in core_ids)
        
        tier1_all_ids = [
            test_id for test_id, test_data in self.report["tests"].items()
            if test_data["tier"].startswith("Tier_1") and test_id != "Test_13"
        ]
        all_numerical_passed = len(tier1_all_ids) > 0 and all(
            self.report["tests"][tid]["status"] == "PASS" for tid in tier1_all_ids
        )
        
        physical_passed = (self.report["tests"].get("Test_12", {}).get("status") == "PASS")
        external_passed = (self.report["tests"].get("Test_13", {}).get("status") == "PASS")

        self.report["tier_1_mandatory_passed"] = all_numerical_passed
        self.report["tier_2_sanity_passed"] = physical_passed
        self.report["tier_3_raw_test_passed"] = False
        self.report["tier_3_validation_passed"] = external_passed
        self.report["executed_at_utc"] = datetime.now(timezone.utc).isoformat()

        statuses = [t["status"] for t in self.report["tests"].values()]
        self.report["summary"] = {
            "total": len(statuses),
            "passed": statuses.count("PASS"),
            "failed": statuses.count("FAIL"),
            "blocked": statuses.count("BLOCKED"),
            "errors": statuses.count("ERROR"),
            "not_implemented": statuses.count("NOT_IMPLEMENTED")
        }

        # Staged Promotion Decision Gate
        if all_numerical_passed and physical_passed and external_passed:
            self.report["M8_REFERENCE_STATUS"] = "VALIDATED_REFERENCE"
            self.report["BASELINE_DEVELOPMENT_ALLOWED"] = True
            self.report["SNAPSHOT_EXPORT_ALLOWED"] = True
        elif all_numerical_passed and physical_passed:
            self.report["M8_REFERENCE_STATUS"] = "VERIFIED_NUMERICALLY_AND_PHYSICALLY"
            self.report["BASELINE_DEVELOPMENT_ALLOWED"] = True
            self.report["SNAPSHOT_EXPORT_ALLOWED"] = False
        elif all_numerical_passed:
            self.report["M8_REFERENCE_STATUS"] = "VERIFIED_NUMERICALLY"
            self.report["BASELINE_DEVELOPMENT_ALLOWED"] = False
            self.report["SNAPSHOT_EXPORT_ALLOWED"] = False
        elif core_passed:
            self.report["M8_REFERENCE_STATUS"] = "CORE_SANITY_TESTS_PASSED"
            self.report["BASELINE_DEVELOPMENT_ALLOWED"] = False
            self.report["SNAPSHOT_EXPORT_ALLOWED"] = False
        else:
            self.report["M8_REFERENCE_STATUS"] = "UNVERIFIED"
            self.report["BASELINE_DEVELOPMENT_ALLOWED"] = False
            self.report["SNAPSHOT_EXPORT_ALLOWED"] = False

        print("=" * 80)
        print(f"OVERALL STATUS: {self.report['M8_REFERENCE_STATUS']}")
        print(f"BASELINE DEV ALLOWED: {self.report['BASELINE_DEVELOPMENT_ALLOWED']}")
        print(f"SNAPSHOT EXPORT ALLOWED: {self.report['SNAPSHOT_EXPORT_ALLOWED']}")
        print("=" * 80)

    def export_report(self, filepath="m8_verification_report.json"):
        with open(filepath, "w") as f:
            json.dump(self.report, f, indent=2)
        print(f"Saved machine-readable verification report to '{filepath}'.")


if __name__ == "__main__":
    runner = M8VerificationRunner()
    runner.run_suite()
    runner.export_report()