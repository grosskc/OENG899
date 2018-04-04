import numpy as np
import scipy.io
import h5py

# Load MAT file
tmp = scipy.io.loadmat("LWIR_HSI_inputs.mat")

# helper function to extract variables stored in matlab structures
struct2dict = lambda struct: {n: struct[n][0, 0].squeeze() for n in struct.dtype.names}

# extract variables from MAT file
X = tmp["X"].flatten().astype(float) # spectral axis, wavenumbers [1/cm]
emis = tmp["emis"].transpose()  # emissivity, spectral dimension first

# atmopspheric state variables
atmos = struct2dict(tmp["atmos"])
P = atmos["P"]*100   # pressure [Pa] from [hPa]
T = atmos["T"]       # temperature profile, [K]
H2O = atmos["H2O"]   # water profile, [ppmv]
O3 = atmos["O3"]*1e6 # ozone profile, [ppmv] from mixing fraction
z = atmos["z"]       # altitude [km]

# atmospheric radiative transfer terms
rad = struct2dict(tmp["rt"])
tau = rad["tau"].transpose()    # transmittance, [no units]
La = 1e6*rad["La"].transpose()  # atmospheric path radiance, [µW/(cm^2 sr cm^{-1})]
Ld = 1e6*rad["Ld"].transpose()  # downwelling radiance, [µW/(cm^2 sr cm^{-1})]

# Reduce file size
nA = tau.shape[1]
nK = 27
ix = np.round(np.linspace(250, nA-251, nK)).astype('uint')
ix1 = np.argsort(tau.mean(0))[ix]
ix2 = np.argsort(La.mean(0))[ix]
ix3 = np.argsort(Ld.mean(0))[ix]
ix = np.unique(np.r_[ix1, ix2, ix3])

tau = tau[:, ix]
La = La[:, ix]
Ld = Ld[:, ix]
T = T[ix,:]
H2O = H2O[ix,:]
O3 = O3[ix,:]

# Overall sort by transmittance
ix = np.argsort(tau.mean(0))
tau = tau[:,ix]
La = La[:, ix]
Ld = Ld[:, ix]
T = T[ix,:]
H2O = H2O[ix,:]
O3 = O3[ix,:]

# Save as HDF5 file
hf = h5py.File('LWIR_HSI_inputs.h5', 'w')
d = hf.create_dataset('X', data=X)
d.attrs['units'] = 'cm^{-1}'
d.attrs['name'] = 'Wavenumbers'
d.attrs['info'] = 'Spectral axis for emis, tau, La, Ld'
d.attrs['label'] = r'$\tilde{\nu} \,\, \left[\si{cm^{-1}} \right]$'

d = hf.create_dataset('emis', data=emis)
d.attrs['units'] = 'none'
d.attrs['name'] = 'Emissivity'
d.attrs['info'] = 'Hemispherically-averaged emissivity'
d.attrs['label'] = r'$\varepsilon(\tilde{\nu})$'

d = hf.create_dataset('tau', data=tau)
d.attrs['units'] = 'none'
d.attrs['name'] = 'Transmissivity'
d.attrs['info'] = 'For nadir-viewing path'
d.attrs['label'] = r'$\tau(\tilde{\nu})$'

d = hf.create_dataset('La', data=La)
d.attrs['units'] = 'µW/(cm^2 sr cm^{-1})'
d.attrs['name'] = 'Atmospheric Path Spectral Radiance'
d.attrs['info'] = 'For nadir-viewing path, earth-to-space'
d.attrs['label'] = r'$L_a(\tilde{\nu})\,\,\left[\si{\micro W/(cm^2.sr.cm^{-1})}\right]$'

d = hf.create_dataset('Ld', data=Ld)
d.attrs['units'] = 'µW/(cm^2 sr cm^{-1})'
d.attrs['name'] = 'Atmospheric Downwelling Spectral Radiance'
d.attrs['info'] = 'Hemispherically-averaged, space-to-earth'
d.attrs['label'] = r'$L_d(\tilde{\nu})\,\,\left[\si{\micro W/(cm^2.sr.cm^{-1})}\right]$'

d = hf.create_dataset('z', data=z)
d.attrs['units'] = 'km'
d.attrs['name'] = 'Altitude'
d.attrs['info'] = 'z=0 at sea level'
d.attrs['label'] = r'$z \,\, \left[ \si{km} \right]$'

d = hf.create_dataset('T', data=T)
d.attrs['units'] = 'K'
d.attrs['name'] = 'Temperature profile'
d.attrs['info'] = ''
d.attrs['label'] = r'$T(z) \,\, \left[ \si{K} \right]$'

d = hf.create_dataset('P', data=P)
d.attrs['units'] = 'Pa'
d.attrs['name'] = 'Pressure profile'
d.attrs['info'] = ''
d.attrs['label'] = r'$P(z) \,\, \left[ \si{Pa} \right]$'

d = hf.create_dataset('H2O', data=H2O)
d.attrs['units'] = 'ppmv'
d.attrs['name'] = 'Water vapor VMR profile'
d.attrs['info'] = 'VMR - volume mixing ratio'
d.attrs['label'] = r'$\mathrm{H_2O}(z)\,\,\left[\mathrm{ppm}_v\right]$'

d = hf.create_dataset('O3', data=O3)
d.attrs['units'] = 'ppmv'
d.attrs['name'] = 'Ozone VMR profile'
d.attrs['info'] = 'VMR - volume mixing ratio'
d.attrs['label'] = r'$\mathrm{O_3}(z)\,\,\left[\mathrm{ppm}_v\right]$'

hf.close()
