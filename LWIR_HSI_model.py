# Import necessary packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py

# Set plotting defaults
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage[adobe-utopia]{mathdesign}, \usepackage{siunitx}'
mpl.rcParams['lines.linewidth'] = 0.5

# Define shortcuts for long LaTeX strings
tnu = r"\tilde{\nu}"  # tilde nu
U_wn = r"\left[\si{cm^{-1}}\right]"  # units wavenumbers
U_rad = r"\left[\si{\micro W/(cm^2.sr.cm^{-1})}\right]"

# turn off interactive plotting
plt.ioff()

# planckian distribution
def planckian(X, T, wavelength=False):
    """
    Compute the Planckian spectral radiance distribution.

    Computes the spectral radiance `L` at wavenumber(s) `X` for a system at
    temperature(s) `T` using Planck's distribution function. `X` must be a scalar
    or a vector. `T` can be of arbitrary dimensions. The shape of output `L` will
    be `(X.size, *T.shape)`.

    Parameters
    ----------
    X : array_like (N,)
      spectral axis, wavenumbers [1/cm], 1D array
    T : array_like
      temperature array, Kelvin [K], arbitrary dimensions
    wavelength : logical
      if true, interprets spectral input `X` as wavelength [micron, µm]

    Returns
    -------
    L : array_like
      spectral radiance in [µW/(cm^2·sr·cm^-1)], or if wavelength=True,
      spectral radiance in [µW/(cm^2·sr·µm)] (microflick, µF)

    Example
    _______
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> X = np.linspace(2000,5000,100)
    >>> T = np.linspace(273,373,10)
    >>> L = planckian(X,T)
    >>> plt.plot(X,L)
    """
    # Physical constants
    # h  = 6.6260689633e-34 # [J s]       - Planck's constant
    # c  = 299792458        # [m/s]       - speed of light
    # k  = 1.380650424e-23  # [J/K]       - Boltzman constant
    # [J m^2 / s] - 1st radiation constant, c1 = 2 * h * c**2
    c1 = 1.19104295315e-16
    # [m K]       - 2nd radiation constant, c2 = h * c / k
    c2 = 1.43877736830e-02

    # Ensure inputs are NumPy arrays
    X = np.asarray(X).flatten()  # X must be 1D array
    T = np.asarray(T)

    # Make X a column vector and T a row vector for broadcasting into 2D arrays
    X = X[:, np.newaxis]
    dimsT = T.shape  # keep shape info for later reshaping into ND array
    T = T.flatten()[np.newaxis, :]

    # Compute Planck's spectral radiance distribution
    # compute using wavelength (with hueristics)
    if wavelength or np.mean(X) < 50:
        if not wavelength:
            print('Assumes X given in µm; returning L in µF')
        X *= 1e-6  # convert to m from µm
        L = c1 / (X**5 * (np.exp(c2 / (X * T)) - 1))  # [W/(m^2 sr m)] SI
        L *= 1e-4  # convert to [µW/(cm^2 sr µm^{-1})]
    else:  # compute using wavenumbers
        X *= 100  # convert to 1/m from 1/cm
        L = c1 * X**3 / (np.exp(c2 * X / T) - 1)  # [W/(m^2 sr m^{-1})]
        L *= 1e4  # convert to [µW/(cm^2 sr cm^{-1})] (1e6 / 1e2)

    # Reshape L if necessary and return
    return np.reshape(L, (X.size, *dimsT))


# plot Planck's distribution over range of atmospheric temperatures
X = np.linspace(10000/12, 10000/8, 500)  # [cm^{-1}], corresponding to 8–12µm
T = np.linspace(250, 310, 5)  # [K]
fig = plt.figure(figsize=(6, 4))
for i, temp in enumerate(T):
    plt.plot(X, planckian(X, temp), label=rf"$T={temp}$ K")
plt.xlabel(rf'Wavenumber, ${tnu}$ ${U_wn}$')
plt.ylabel(rf'Blackbody Radiance, $B({tnu},T)$ ${U_rad}$')
plt.legend()
fig.tight_layout()
fig.savefig('figures/Planckian.png', dpi=300)

# Load H5 file
f = h5py.File("LWIR_HSI_inputs.h5", "r")
print(list(f.keys()))

# Extract spectral axis, wavenumbers, (nX,)
X = f["X"][...]

# example demonstrating that there is metadata in this HDF5 file
print(f"""The spectral axis, {f["X"].attrs['name']}, has units """ +
      f"""{f["X"].attrs['units']} and spans {X.min():0.1f} ≤ X ≤ {X.max():0.1f}""")

# Extract emissivity, (nX, nM)
emis = f["emis"][...]  # spectral dimension first

# atmospheric state variables, (nA, nZ)
z = f["z"][...]      # altitude above sea level, [km]
Tz = f["T"][...]     # temperature profile, [K]
Ts = Tz[:, 0]        # surface temperature, [K]
H2O = f["H2O"][...]  # water vapor volume mixing fraction, [ppm]
O3 = f["O3"][...]    # ozone volume mixing fraction, [ppm]

# atmospheric radiative transfer terms
tau = f["tau"][...]  # transmittance, [no units]
La = f["La"][...]    # atmospheric path radiance, [µW/(cm^2 sr cm^{-1})]
Ld = f["Ld"][...]    # downwelling radiance, [µW/(cm^2 sr cm^{-1})]

# close H5 file
f.close()

# take the first, middle, and last atmospheric states (which have been sorted)
# byt the spectrally-averaged transmittance
nA = Tz.shape[0]
ixA = np.linspace(0, nA-1, 3).astype('int')
fig = plt.figure(figsize=(7.5, 10))

# temperature
plt.subplot(2, 2, 1)
for a in ixA:
    plt.semilogy(Tz[a, :].transpose(), z, label=fr'atmID\#{a}')
plt.xlabel('Temperature, T [K]')
plt.ylabel('Altitude, $z$ [km]')
plt.legend()

# mixing fraction profiles
plt.subplot(2, 2, 3)
for i, a in enumerate(ixA):
    c = f"C{i:d}"
    h2o = H2O[a, :].transpose()
    o3 = O3[a, :].transpose()
    plt.loglog(h2o, z, color=c, label=fr'H2O, atmID\#{a}')
    plt.loglog(o3, z, '--', color=c, label=fr'O3, atmID\#{a}')
plt.xlabel('Mixing Fraction [ppmv]')
plt.ylabel('Altitude, z [km]')
plt.legend()

# transmittance
plt.subplot(2, 2, 2)
for a in ixA:
    plt.plot(X, tau[:, a], label=fr'atmID\#{a}')
plt.xlabel(rf'Wavenumber, ${tnu}$ ${U_wn}$')
plt.ylabel(rf'Transmittance, $\tau({tnu})$')
plt.legend()

# path radiance
plt.subplot(2, 2, 4)
for a in ixA:
    plt.plot(X, La[:, a], label=fr'atmID\#{a}')
plt.xlabel(rf'Wavenumber, ${tnu}$ ${U_wn}$')
plt.ylabel(rf'Path Radiance, $L_a({tnu})$ ${U_rad}$')
plt.legend()
fig.tight_layout()
fig.savefig('figures/AtmosInputsOutputs.png', dpi=300)

# function to compute apparent radiances efficiently
def compute_radiance(X, emis, Ts, tau, La, Ld, dT=None):
    r"""
    Compute spectral radiance for given emissivities and atmospheric states.

    Efficienetly computes (via broadcasting) every combination of spectral radiance
    for a set of emissivity profiles, a set of atmospheric radiative terms, and an
    optional range of surface temperatures.

    Parameters
    __________
    X: array_like (nX,)
      spectral axis in wavenumbers [1/cm], 1D array of length `nX`
    emis: array_like (nX, nE)
      emissivity array – `nE` is the number of materials
    Ts: array_like (nA,)
      surface temperature [K], 1D array of length `nA`
    tau: array_like (nX, nA)
      atmospheric transmittance between source and sensor [0 ≤ tau ≤ 1]
    La: array_like (nX, nA)
      upwelling atmospheric path radiance [µW/(cm^2 sr cm^{-1})]
    Ld: array_like (nX, nA)
      hemispherically-averaged atmospheric downwelling radiance [µW/(cm^2 sr cm^{-1})]
    dT: array_like (nT,), optional {None}
      surface temperature deltas, relative to `Ts` [K]

    Returns
    _______
    L: array_like (nX, nE, nA) or (nX, nE, nA, nT)
      apparent spectral radiance
    """
    if dT is not None:
        T_ = Ts.flatten()[:, np.newaxis] + \
            np.asarray(dT).flatten()[np.newaxis, :]
        B_ = planckian(X, T_)[:, np.newaxis, :]
        tau_ = tau[:, np.newaxis, :, np.newaxis]
        La_ = La[:, np.newaxis, :, np.newaxis]
        Ld_ = Ld[:, np.newaxis, :, np.newaxis]
        em_ = emis[:, :, np.newaxis, np.newaxis]
    else:
        T_ = Ts.flatten()
        B_ = planckian(X, T_)[:, np.newaxis, :]
        tau_ = tau[:, np.newaxis, :]
        La_ = La[:, np.newaxis, :]
        Ld_ = Ld[:, np.newaxis, :]
        em_ = emis[:, :, np.newaxis]
    L = tau_ * (em_ * B_ + (1-em_) * Ld_) + La_
    return L

# Compute radiance for given emis and atmos rad txfr inputs
L = compute_radiance(X, emis, Ts, tau, La, Ld)

# helper plotting tool
def plot_apparent_rad(eID=[0], aID=[0], k=0):
    if len(eID) > 1:
        aID = aID[0]
        for i, e in enumerate(eID):
            plt.plot(X, L[:, e, aID], label=f"k={i}, Matl ID = {eID[i]}")
        plt.title(fr'Atm ID \# {aID}')
    else:
        for i, a in enumerate(aID):
            plt.plot(X, L[:, eID, a], label=fr"Atm ID \#{aID[i]}")
            plt.title(f'k={k}, Material ID = {eID}')
    plt.xlabel(rf'${tnu}$ ${U_wn}$')
    plt.ylabel(rf"$L_{{o,k}}({tnu})$ ${U_rad}$")
    plt.legend()
    return None

# take uniform sampling of atm
def aIDs(N): return np.linspace(0, tau.shape[1]-1, N).astype('uint')

# sort emissivities by mean emissivity -- most emissivities in this database
# are high, so take first two plus a high one
ix_em = np.argsort(emis.mean(axis=0))
eIDs = ix_em[[0, 1, -2]]

# set up plot parameters
N_atm = 3
N_em = len(eIDs)

# Plot each material separately, while showing apparent radiance under a common
# set of distinct atmospheres
fig = plt.figure(figsize=(8, 10))
for i, e in enumerate(eIDs):
    ax = plt.subplot(N_em, 1, i+1)
    plot_apparent_rad(eID=[e], aID=aIDs(N_atm), k=i)
    ax2 = ax.twinx()
    plt.plot(X, emis[:, e], color='black', label='Emissivity')
    plt.ylabel('Emissivity')
fig.tight_layout()
fig.savefig('figures/AtmosphericVariability.png', dpi=300)

# Plot each atmospheric state separately, while showing apparent radiance under
# a common set of distinct materials
fig = plt.figure(figsize=(8, 10))
for i, a in enumerate(aIDs(N_atm)):
    ax = plt.subplot(N_atm, 1, i+1)
    plot_apparent_rad(eID=eIDs, aID=[a])
fig.tight_layout()
fig.savefig('figures/EmissivityVariability.png', dpi=300)

# clean up Ld due to division by small number
ix = tau < 1e-4
Ld_ = np.copy(Ld)
Ld_[ix] = np.nan

# Plot apparent spectral radiance and various radiative transfer terms
def plot_radiance(atmID=0, emisID=0):
    fig = plt.figure(figsize=(8.0, 10.0))
    # 1st plot - apparent radiance
    plt.subplot(2, 2, 1)
    lbl = f"Atmos \#{atmID}, Matl \#{emisID}"
    plt.plot(X, L[:, emisID, atmID], label=lbl)
    plt.ylabel(rf"$L({tnu})$ ${U_rad}$")
    plt.xlabel(rf'${tnu}$ ${U_wn}$')
    plt.legend()

    # 2nd plot - atmospheric radiation terms
    ax1 = plt.subplot(2, 2, 2)
    a1 = ax1.plot(X, tau[:, atmID], color='C0', label=r'$\tau$')
    plt.ylabel(rf"$\tau({tnu})$")
    plt.xlabel(rf'${tnu}$ ${U_wn}$')
    ax1.legend()
    ax2 = ax1.twinx()
    a2 = ax2.plot(X, La[:, atmID], color='C1', label="$L_a$")
    a3 = ax2.plot(X, Ld_[:, atmID], color='C2', label="$L_d$")
    plt.ylabel(rf'$L_{{a,d}}({tnu})$ ${U_rad}$')
    leg = a1+a2+a3
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    fig.tight_layout()

    # 3rd plot - surface-leaving radiance and reflectivity
    ax1 = plt.subplot(2, 2, 3)
    B = planckian(X, Ts[atmID])
    a1 = plt.plot(X, B, label=f"Planckian, T={Ts[atmID]:0.1f} K")
    a2 = plt.plot(X, emis[:, emisID] * B, label="Thermal Emission")
    a3 = plt.plot(X, (1 - emis[:, emisID]) * Ld_[:, atmID], label="Reflected")
    plt.xlabel(rf'${tnu}$ ${U_wn}$')
    plt.ylabel(rf'$L_s({tnu})$ ${U_rad}$')
    ax2 = ax1.twinx()
    a4 = plt.plot(X, emis[:, emisID], color='C3', label="Emissivity")
    plt.ylabel('Emissivity')
    leg = a1+a2+a3+a4
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)

    # 4th plot - atmospheric state variables
    # trim atmospheric profiles to the first 17 km
    ix = z <= 17
    z_ = z[ix]
    H2O_ = H2O[atmID, ix]
    Tz_ = Tz[atmID, ix]
    ax1 = plt.subplot(2, 2, 4)
    b1 = ax1.plot(Tz_, z_, color="C0", label="Temperature")
    plt.xlabel('Temperature [K]')
    plt.ylabel('Altitude [km]')
    ax2 = ax1.twiny()
    b2 = ax2.plot(H2O_, z_, color="C1", label=r"$\mathrm{H_2O}$")
    plt.xlabel('Mixing Fraction [ppmv]')
    leg = b1+b2
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    fig.tight_layout()
    return fig


# loop over each material and atmospheric state
for a in aIDs(N_atm):
    for e in eIDs:
        f = plot_radiance(atmID=a, emisID=e)
        f.savefig(f"figures/RadOverview-aID{a:03d}-eID{e:03d}", dpi=300)
