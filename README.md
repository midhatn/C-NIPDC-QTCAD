# C-NIPDC-QTCAD
**A verification-first 1D quantum-transport framework for resonant tunneling devices using self-consistent NEGF-Poisson simulation.**
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.15%2B-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Status](https://img.shields.io/badge/status-verified%20numerically-blue)](#verification-status)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
> **Current qualification level:** `VERIFIED_NUMERICALLY`  
> Tier 1 numerical verification has passed. Tier 2 NDR sanity and Tier 3 matched external validation remain gated. Baseline development and formal reference-snapshot export are therefore disabled.
---
## Overview
C-NIPDC-QTCAD is a research-oriented quantum-device simulation and verification framework for one-dimensional resonant tunneling structures. The current implementation couples:
- a variable-effective-mass finite-difference Hamiltonian;
- analytical retarded surface Green functions for semi-infinite contacts;
- ballistic nonequilibrium Green-function transport;
- transverse Fermi supply integration;
- self-consistent Poisson electrostatics;
- continuation, damping, and safeguarded mixing;
- numerical, physical, and external-validation governance gates.
The central design principle is **verification before promotion**. A model is not treated as a baseline merely because it produces an I-V curve. It must first pass explicit tests for equilibrium, gauge invariance, current conservation, mesh convergence, energy convergence, nonlinear residual closure, and continuation consistency.
This repository is intended for research, diagnostics, reproducibility, and solver development. It is not yet a validated production TCAD package.
---
## Key capabilities
### Quantum transport
- One-dimensional ballistic NEGF formulation.
- BenDaniel-Duke variable-mass discretization.
- Analytical contact surface Green functions with retarded-branch selection.
- Contact self-energies and broadening functions.
- Transmission, carrier density, terminal current, and bond-current evaluation.
- Transverse supply function with finite-temperature Fermi statistics.
- Composite energy quadrature clustered around both contact band edges.
### Self-consistent electrostatics
- Interior sparse Poisson operator with spatially varying permittivity.
- Dirichlet contact boundary conditions.
- Backward-error-scaled Poisson residual.
- Conjunctive convergence requirement:
  \[
  r_{\mathrm{SC}} < \varepsilon_{\mathrm{SC}},
  \qquad
  \|\Delta\phi\|_{\infty} < \varepsilon_{\phi},
  \qquad
  r_{\mathrm{BC}} < \varepsilon_{\mathrm{BC}}.
  \]
- Damped Gummel iteration.
- Safeguarded two-history Anderson-style acceleration.
- Continuation with recursive voltage bisection.
### Verification and governance
The verification harness is divided into three tiers:
1. **Tier 1: Numerical verification**
2. **Tier 2: Physical sanity**, including converged qualitative NDR
3. **Tier 3: Matched external validation against a published device**
Formal baseline development and snapshot export are controlled by machine-readable promotion gates.
---
## Verification status
The latest recorded verification state is:
```text
M8_REFERENCE_STATUS:           VERIFIED_NUMERICALLY
Tier 1 mandatory tests:        PASS
Tier 2 physical sanity:        NOT PASSED
Tier 3 external validation:    BLOCKED
Baseline development allowed:  False
Snapshot export allowed:       False
```
### Tier 1 results

| Test | Description | Status | Recorded metric | Tolerance |
| :--- | :--- | :--- | :--- | :--- |
| Test 01 | Fermi inversion sanity | PASS | `2.967e-12` | `1.000e-09` |
| Test 02 | Surface-GF branch selection | PASS | `0.000e+00` | `5.000e-01` |
| Test 03 | Uniform passband transmission | PASS | `3.186e-14` | `1.000e-04` |
| Test 04 | Exact gauge invariance | PASS | `6.284e-12` | `1.000e-08` |
| Test 05 | Bond-current/Landauer consistency | PASS | `9.236e-14` | `1.000e-04` |
| Test 06 | Preliminary zero-bias equilibrium | PASS | `6.809e-03` | `2.000e-02` |
| Test 07 | Variable-mass barrier benchmark | PASS | `4.119e-03` | `5.000e-02` |
| Test 08a | Prescribed-potential mesh check | PASS | `1.860e-02` | `3.000e-02` |
| Test 08b | Self-consistent I-V mesh convergence | PASS | `3.608e-03` | `5.000e-02` |
| Test 09 | Charge/current energy convergence | PASS | `1.667e-03` | `5.000e-03` |
| Test 10 | Coupled interior residual sweep | PASS | `4.638e-05` | `1.000e-04` |
| Test 11 | Forward/reverse continuation consistency | PASS | `7.834e-04` | `1.000e-03` |

### Remaining qualification work
- **Test 12:** qualitative NDR detection remains incomplete because the fixed-voltage self-consistent solve loses convergence in the high-bias resonance transition.
- **Test 13:** matched literature validation remains blocked until Tier 2 passes.
- A dedicated diagnostic localized the current fixed-point stability boundary to the interval between `0.256 V` and `0.257 V` for the present model and solver settings.
- Energy-resolution sensitivity at `0.257 V` is being tested with requested grids of 320, 640, and 1280 points before introducing more advanced continuation methods.
Failed nonlinear iterates are never accepted as physical I-V points or used for PVCR estimation.
---
## Repository structure
The exact tree may evolve, but the active verification workflow uses the following files:
```text
C-NIPDC-QTCAD/
├── m8_verifier.py
├── diagnose_ndr_026.py
├── diagnose_ndr_fine.py
├── diagnose_ndr_energy_resolution.py
├── m8_execution.log
├── m8_verification_report.json
├── ndr_boundary_trace.json
├── ndr_boundary_summary.csv
├── ndr_energy_resolution_trace.json
├── ndr_energy_resolution_summary.csv
├── ndr_reference_0256.npz
├── LICENSE
├── README.md
└── requirements.txt
```
### Main files
- `m8_verifier.py`: NEGF-Poisson solver, verification tests, and promotion governance.
- `diagnose_ndr_026.py`: focused continuation diagnostic around the original `0.260 V` stall.
- `diagnose_ndr_fine.py`: detailed comparison of fixed damping, adaptive damping, and safeguarded Anderson mixing near the high-bias boundary.
- `diagnose_ndr_energy_resolution.py`: 320/640/1280-point energy-resolution study at `0.257 V`, initialized from one common converged `0.256 V` state.
- `m8_verification_report.json`: machine-readable qualification report.
- `ndr_*_trace.json`: per-iteration diagnostic telemetry.
- `ndr_*_summary.csv`: compact diagnostic summaries.
> Generated logs, traces, reports, and NPZ files may be excluded from source control if they are large or machine-specific. If excluded, document the exact command and environment required to reproduce them.
---
## Installation
### Requirements
- Python 3.12 or newer recommended
- NumPy
- SciPy
Create and activate a virtual environment:
```powershell
cd C:\Projects\QTCAD
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy
```
Alternatively, with Conda:
```powershell
conda create -n qtcad python=3.12 numpy scipy -y
conda activate qtcad
```
If a `requirements.txt` file is present:
```powershell
python -m pip install -r requirements.txt
```
---
## Quick start
Run the complete verification suite:
```powershell
python m8_verifier.py
```
Expected machine-readable output:
```text
m8_verification_report.json
```
The console summary reports the qualification state and whether baseline development or snapshot export is permitted.
---
## Diagnostic workflows
### Fine nonlinear-boundary diagnostic
```powershell
python diagnose_ndr_fine.py
```
Expected outputs:
```text
ndr_boundary_trace.json
ndr_boundary_summary.csv
```
This diagnostic compares:
- Configuration A: fixed damping, `alpha = 0.02`;
- Configuration B: adaptive damping, `alpha in [0.005, 0.08]`;
- Configuration C: dual-criterion candidate-safeguarded Anderson mixing.
The cycle indicator is
\[
R_2 =
\frac{
\left\langle \|\phi_k-\phi_{k-2}\|_\infty \right\rangle
}{
\left\langle \|\phi_k-\phi_{k-1}\|_\infty \right\rangle
}.
\]
A small value of `R2` alone is not sufficient to declare a period-two orbit. The implementation also requires persistent nonconvergence, a nonvanishing one-step norm, enough samples, and a fixed-point defect above tolerance.
### Energy-resolution diagnostic
Keep the following files under distinct names:
```text
diagnose_ndr_fine.py
diagnose_ndr_energy_resolution.py
```
Then run:
```powershell
python diagnose_ndr_energy_resolution.py
```
Expected outputs:
```text
ndr_reference_0256.npz
ndr_energy_resolution_trace.json
ndr_energy_resolution_summary.csv
```
The script constructs one converged `0.2560 V` reference potential, hashes the reference state, shifts an identical copy to satisfy the `0.2570 V` boundary, and compares requested energy-grid resolutions of 320, 640, and 1280 points.
> Do not save the energy-resolution script as `diagnose_ndr_fine.py`. It imports helper functions from that module, and using the same filename creates a circular import.
---
## Reproducibility
For every reported run, retain:
- Python, NumPy, and SciPy versions;
- operating-system information;
- spatial spacing and node count;
- energy quadrature settings;
- voltage path and continuation policy;
- nonlinear tolerances and iteration limit;
- current, well charge, residual, and fixed-point defect;
- mixing parameters and fallback counts;
- source commit hash;
- generated JSON or CSV report.
A typical environment record is available inside `m8_verification_report.json`.
For a publication-quality result, create a clean environment, record the Git commit, run the full verification suite, and archive the report and terminal log together.
---
## Numerical model summary
The finite-difference Hamiltonian uses the variable-mass BenDaniel-Duke form. Contact effects enter through retarded self-energies:
\[
\Sigma_\alpha^r(E) = \tau_\alpha^\dagger g_\alpha^r(E)\tau_\alpha,
\qquad
\Gamma_\alpha(E) = i\left(\Sigma_\alpha^r-\Sigma_\alpha^a\right).
\]
Transmission is evaluated using
\[
T(E)=\operatorname{Tr}
\left[
\Gamma_L G^r \Gamma_R G^a
\right].
\]
The electrostatic problem is coupled self-consistently through
\[
-\nabla\cdot\left(\varepsilon\nabla\phi\right)
= q\left(N_D-n[\phi]\right).
\]
The relaxed fixed-point iteration is
\[
\phi^{(k+1)}=
(1-\alpha)\phi^{(k)}+
\alpha\mathcal{P}\!\left(\mathcal{N}(\phi^{(k)})\right).
\]
The implementation does not accept convergence based only on a normalized Poisson residual. The fixed-point map defect and boundary residual must also satisfy their declared tolerances.
---
## Governance model
The repository intentionally distinguishes four states:
```text
UNVERIFIED
CORE_SANITY_TESTS_PASSED
VERIFIED_NUMERICALLY
VERIFIED_NUMERICALLY_AND_PHYSICALLY
VALIDATED_REFERENCE
```
Promotion rules are strict:
- Tier 1 pass permits the label `VERIFIED_NUMERICALLY`.
- Tier 1 and Tier 2 pass permit baseline development.
- Tier 3 matched validation is required for `VALIDATED_REFERENCE`.
- Formal snapshot export is prohibited until all three tiers pass.
Do not weaken test thresholds merely to promote a result. A blocked or failed test is an engineering outcome, not an inconvenience to suppress.
---
## Known limitations
- The current model is one-dimensional and ballistic.
- Scattering and dissipative self-energies are not yet included.
- Tier 2 high-bias NDR convergence is unresolved.
- Tier 3 source-matched experimental validation is not yet implemented.
- Series resistance and measurement de-embedding are not yet part of a matched benchmark.
- The current high-bias solver can become noncontractive near rapid resonance redistribution.
- Passing numerical verification does not establish experimental predictive accuracy.
---
## Roadmap
- [x] Correct contact broadening dimensions.
- [x] Correct transverse supply asymptotics.
- [x] Add dual-edge composite energy quadrature.
- [x] Add mesh and energy convergence tests.
- [x] Add backward-error and conjunctive convergence checks.
- [x] Pass forward/reverse continuation consistency.
- [x] Reach Tier 1 `VERIFIED_NUMERICALLY` status.
- [ ] Complete 320/640/1280 energy-resolution study at `0.257 V`.
- [ ] Determine whether the high-bias instability is quadrature-driven or operator-driven.
- [ ] Introduce pseudo-transient or Newton-Krylov continuation if required.
- [ ] Complete a fully converged NDR sweep and PVCR sanity test.
- [ ] Implement one exact literature-matched RTD benchmark.
- [ ] Add automated CI for fast tests and scheduled extended verification.
- [ ] Publish versioned reproducibility artifacts.
---
## Contributing
Contributions are welcome after the repository establishes contribution and licensing policies. Useful contributions include:
- numerical-analysis reviews;
- additional analytical benchmarks;
- independent reproduction of verification results;
- improved energy quadrature;
- robust nonlinear solvers;
- source-matched RTD benchmark definitions;
- documentation and test coverage.
Recommended workflow:
1. Fork the repository.
2. Create a focused branch.
3. Add or update tests with the implementation.
4. Run the verification suite.
5. Include the generated report and environment details in the pull request.
6. Avoid changing qualification thresholds without a documented numerical justification.
---
## Citation
A formal citation has not yet been published. Until a release and archival DOI are available, cite the repository and the exact commit used:
```bibtex
@software{cnipdc_qtcad,
  author  = {Nashar, Midhat},
  title   = {C-NIPDC-QTCAD: Verification-First NEGF-Poisson Quantum Transport Framework},
  url     = {[https://github.com/midhatn/C-NIPDC-QTCAD](https://github.com/midhatn/C-NIPDC-QTCAD)},
  note    = {Cite the exact Git commit and access date},
  year    = {2026}
}
```
---
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
---
## Disclaimer
This software is a research prototype. Numerical convergence does not by itself establish physical validity, fabrication readiness, safety, or experimental agreement. Verify all assumptions, units, boundary conditions, discretizations, and benchmark definitions before using results in research publications or engineering decisions.
