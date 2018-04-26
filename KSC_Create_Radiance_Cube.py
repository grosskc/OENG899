import numpy as np
import scipy.io as sio
import h5py
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import os

# Set plotting defaults
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage[adobe-utopia]{mathdesign}, \usepackage{siunitx}'
mpl.rcParams['lines.linewidth'] = 0.5

# Define shortcuts for long LaTeX strings
tnu = r"\tilde{\nu}"  # tilde nu
U_wn = r"\left[\si{cm^{-1}}\right]"  # units wavenumbers
U_rad = r"\left[\si{\micro W/(cm^2.sr.cm^{-1})}\right]"

def select_atmosphere(temp_param_list):
    """
    Select one atmosphere from LWIR_HSI_inputs.mat.

    This function selects one atmosphere from LWIR_HSI_inputs.mat.  The atmosphere with the closest surface temperature to surface_temp is used, where T[:, 0] is the assumed surface temperature for each atmosphere in the database.

    Parameters
    ----------
    temp_param_list : List length 2 (float)
      temperature, [K] followed by the temperature variance

    Returns
    -------
    local_T: Local temperature profile, [K]
    local_La: Local atmospheric path radiance, [µW/(cm^2 sr cm^{-1})]
    local_Ld: Local downwelling radiance, [µW/(cm^2 sr cm^{-1})]
    local_tau: Local transmittance, [no units]
    emis: Emissivity
    X: spectral axis, wavenumbers [1/cm]
    """
    surface_temp = temp_param_list[0]
    # Load MAT file
    tmp = sio.loadmat("data/LWIR_HSI_inputs.mat")

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

    local_T_idx = (np.abs(T[:,0]-surface_temp)).argmin()
    print("Closest Surface Temp is {} K".format(T[local_T_idx, 0]))

    atmospheric_index = np.int(np.min(local_T_idx))

    local_P = P
    local_T = T[atmospheric_index, :]
    local_H20 = H2O[atmospheric_index, :]
    local_O3 = O3[atmospheric_index, :]
    local_z = z
    local_La = La[:,atmospheric_index]
    local_Ld = Ld[:, atmospheric_index]
    local_tau = tau[:, atmospheric_index]

    # Pack up all the variables into atmosphere list
    local_atmosphere = [X, emis, local_P, local_T, local_H20, local_O3, local_z, local_La, local_Ld, local_tau]

    return local_atmosphere


def assign_label_emis(class_labels, em_):
    """
    Assigns emissivities to labeled pixels.

    This function creates a dictionary mapping class labels to emissivities.  This currently doesn't consider material properties and just takes the first number of required emissivities from the em_ variable in LWIR_HSI_inputs.mat.

    Parameters
    ----------
    class_labels : List (strings)
    
    em_ : List (float)
    emissivities from the LWIR_HSI_inputs.mat database

    Returns
    -------
    new_em_: Dict {class_name: emissivity}
    Mapping from pixel label strings to emissivity profiles
    """
    new_em_ = {}
    for i in range(len(class_labels)):
        new_em_[class_labels[i]] = em_[:,i]
    return(new_em_)


def create_emis_data(em_dict, labeled_data, class_labels, all_pixels=True):
    """
    Uses the result of assign_label_emis to create a data cube where the first two axes are the spatial coordinates and third axis is the emissivitiy at all bands.

    Parameters
    ----------
    em_dict : Dictionary {class_name: emissivity}

    labeled_data : 2D Array (int)
        Contains numeric class assignment for each pixel
    class_labels : List (strings)
        Names corresponding to numeric class assignments in labeled_data
    all_pixels : boolean
        Placeholder for only using the originally labeled pixels

    Returns
    ----------
    emis_cube : 2D Array (float64)
        Contains emissivities for every pixel.  Spatial dimensions are flattened.
    """
    
    emis_cube = []
    if all_pixels:
        assert labeled_data.shape[1] > 1, "Incorrect labeled data shape"
        for i in range(labeled_data.shape[0]):
            for j in range(labeled_data.shape[1]):
                emis_cube.append(em_dict[class_labels[np.int(labeled_data[i,j])-1]])
        emis_cube = np.asarray(emis_cube)
    else:
        flattened_label = np.asarray(labeled_data.reshape(labeled_data.size))
        # Extract labeled pixels for training
        labeled_pixel_index = (flattened_label > 0)
        labeled_data = flattened_label[labeled_pixel_index]
        for i in range(labeled_data.shape[0]):
            emis_cube.append(em_dict[class_labels[labeled_data[i]-1]])
        emis_cube = np.asarray(emis_cube)
    
    return emis_cube


def assign_pixel_temps(num_labels, temp_param_list):
    """
    Assigns a surface temperature to each pixel.  The mean temperature when the KSC data was collected was 287 K.  The function randomly assigns temperatures.  More work is needed on intelligently assigning these temps. 

    Parameters
    ----------
    num_labels : scalar (int)
        The number of pixels to assign temperatures to
    temp_param_list : List length 2 (float)
      temperature [K], followed by the temperature variance for normally distributed pixel temperatures

    Returns
    ----------
    pix_temp : array of floats (N,)
        The temperature for each pixel in a flattened array
    """
    surface_temp = temp_param_list[0]
    var_temp = temp_param_list[1]

    # Random assignment
    pix_temp = []
    for i in range(num_labels):
        rnd = random.random()
        pix_temp.append((rnd * var_temp) + surface_temp)
    return np.asarray(pix_temp)


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
    c1 = 1.19104295315e-16  # [J m^2 / s] - 1st radiation constant, c1 = 2 * h * c**2
    c2 = 1.43877736830e-02  # [m K]       - 2nd radiation constant, c2 = h * c / k

    # Ensure inputs are NumPy arrays
    X = np.asarray(X).flatten()  # X must be 1D array
    T = np.asarray(T)

    # Make X a column vector and T a row vector for broadcasting into 2D arrays
    X = X[:, np.newaxis]
    dimsT = T.shape  # keep shape info for later reshaping into ND array
    T = T.flatten()[np.newaxis, :]

    # Compute Planck's spectral radiance distribution
    if wavelength or np.mean(X) < 50:  # compute using wavelength (with hueristics)
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


def get_all_pixel_radiances(atmos, labeled_data, class_labels, temp_param_list, all_pixels=True):
    """
    Creates hyperspectral data cube using predefined pixel label map.  The pixel labels correspond to emissivities that are used with the atmospheric parameters and surface temps to create unique spectra for each pixel.

    Parameters
    ----------
    atmos : List
        Contains spectral bands, emissivities for all pixels, and atmospheric parameters: P, T, H20, O3, z, La, Ld, tau

    labeled_data : 2D Array (int)
        Contains numeric class assignment for each pixel

    class_labels : List (strings)

    all_pixels : boolean
        Placeholder for only using the originally labeled pixels

    temp_param_list : List length 2 (float)
      temperature [K], followed by the temperature variance for normally distributed pixel temperatures
    
    Returns
    ----------
    L : 2D Array (float64)
        Spectral dimension first followed by spatial dimensions flattened

    """
    start = time.time()
    X, emis, local_P, local_T, local_H20, local_O3, local_z, local_La, local_Ld, local_tau = atmos
    local_tau = local_tau[:, np.newaxis]
    local_Ld = local_Ld[:, np.newaxis]
    local_La = local_La[:, np.newaxis]

    if all_pixels:
        num_pixels = labeled_data.shape[0] * labeled_data.shape[1]
        truth_map = np.genfromtxt('data/pixel_predictions.csv', dtype=float, delimiter=",")
    else:
        num_pixels = np.sum(labeled_data > 0)  # 0 represents an unlabeled pixel
        truth_map = labeled_data
    
    print(".........Assigning Pixel Temperatures.........")

    pixel_temps = assign_pixel_temps(num_pixels, temp_param_list)

    print(".........Assigning Pixel Emissivities.........")

    ems_dict = assign_label_emis(class_labels, emis)
    em_ = np.transpose(create_emis_data(ems_dict, truth_map, class_labels, all_pixels))

    print(".........Calculating Planckian.........")

    B_ = planckian(X, pixel_temps)

    print(".........Calculating Radiances.........")

    # L_array = np.asarray(L)
    # em_.shape = (526, 314368)
    # B_.shape = (526, 314368)
    # local_tau.shape = (526, 1)
    # local_Ld.shape = (526, 1)
    # local_La.shape = (526, 1)
    L = local_tau * (em_ * B_ + (1-em_) * local_Ld) + local_La
    
    stop = time.time()
    print("Done Creating Hyperspectral Cube Total Time: {} seconds".format(np.int(stop-start)))

    return L


def plot_augmented_radiance(radiance_cube, atmos, class_labels, class_to_plot):
    """
    This function plots created radiance data for specific types of pixel labels.

    Parameters
    ----------
    radiance_cube : 2D Array (float64)
        Spectral dimension first followed by spatial dimensions flattened.  This is the array returned from get_all_pixel_radiances

    atmos : List
        Contains all atmospheric objects such as wavenumbers, emissivities, P, T, H2O, O3, z, La, Ld, tau
    
    class_labels : List (strings)
        Names corresponding to numeric class assignments in labeled_data

    class_to_plot : string
        Name of the class (Water, Scrub, etc) that plots are required for

    Returns
    ----------
    Plots are saved to figures/augmented_radiance_{}.png where the class_to_plot will be in the brackets. 
    """
    print("Creating plots for radiance, transmittance, downwelling and path radiance, planckian, temperature and H2O")
    radiance_cube = np.reshape(radiance_cube.T, [gt_data.shape[0], gt_data.shape[1], L.shape[0]])

    pixel_labels = np.genfromtxt('data/pixel_predictions.csv', dtype=float, delimiter=",")

    X, emis, local_P, local_T, local_H2O, local_O3, local_z, local_La, local_Ld, local_tau = atmos

    fig = plt.figure(figsize=(8.0, 10.0))
    # 1st plot - apparent radiance
    plt.subplot(2, 2, 1)
    label_id = class_labels.index(class_to_plot) + 1
    class_pixels = (pixel_labels == label_id)
    flatten_pix_id = class_pixels.flatten()
    flattened_cube = np.reshape(radiance_cube, [radiance_cube.shape[0] * radiance_cube.shape[1], radiance_cube.shape[2]])

    plot_spectra = flattened_cube[flatten_pix_id, :]
    pixel_emis = emis[:, label_id -1]
    
    # How to select which pixels to plot?  Currently just picking pixel 0.
    plt.plot(X.T, plot_spectra[0,:].T, linewidth=0.5)
    plt.title(class_to_plot)
    plt.xlabel(rf'Wavenumber, ${tnu}$ ${U_wn}$')
    plt.ylabel(rf"$L({tnu})$ ${U_rad}$")

    # 2nd plot - atmospheric radiation terms
    ax1 = plt.subplot(2, 2, 2)
    a1 = ax1.plot(X, local_tau, color='C0', label=r'$\tau$')
    plt.ylabel(rf"$\tau({tnu})$")
    plt.xlabel(rf'${tnu}$ ${U_wn}$')
    ax1.legend()
    ax2 = ax1.twinx()
    a2 = ax2.plot(X, local_La, color='C1', label="$L_a$")
    a3 = ax2.plot(X, local_Ld, color='C2', label="$L_d$")
    plt.ylabel(rf'$L_{{a,d}}({tnu})$ ${U_rad}$')
    leg = a1+a2+a3
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    fig.tight_layout()

    # 3rd plot - surface-leaving radiance and reflectivity
    ax1 = plt.subplot(2, 2, 3)
    B = planckian(X, local_T[0])
    a1 = plt.plot(X, B, label=f"Planckian, T={local_T[0]:0.1f} K")
    a2 = plt.plot(X, pixel_emis * B, label="Thermal Emission")
    a3 = plt.plot(X, (1 - pixel_emis) * local_Ld, label="Reflected")

    plt.xlabel(rf'${tnu}$ ${U_wn}$')
    plt.ylabel(rf'$L_s({tnu})$ ${U_rad}$')
    ax2 = ax1.twinx()
    a4 = plt.plot(X, pixel_emis, color='C3', label="Emissivity")
    plt.ylabel('Emissivity')
    leg = a1+a2+a3+a4
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)

    # 4th plot - atmospheric state variables
    # trim atmospheric profiles to the first 17 km
    ix = local_z <= 17
    local_z = local_z[ix]
    T = local_T[ix]
    local_H2O = local_H2O[ix]
    ax1 = plt.subplot(2, 2, 4)
    b1 = ax1.plot(T, local_z, color="C0", label="Temperature")
    plt.xlabel('Temperature [K]')
    plt.ylabel('Altitude [km]')
    ax2 = ax1.twiny()
    b2 = ax2.plot(local_H2O, local_z, color="C1", label=r"$\mathrm{H_2O}$")
    plt.xlabel('Mixing Fraction [ppmv]')
    leg = b1+b2
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    fig.tight_layout()
    print("Saving Figure for {0} to: figures/augmented_radiance_{0}.png".format(class_to_plot))
    plt.savefig("figures/augmented_radiance_{}.png".format(class_to_plot))


def save_data_cube(radiance_cube, atmos, temp_param_list, labeled_data, save_path):
    """
    This function saves the data cube generated by get_all_pixel_radiances.  Each cube is 1.3GB.  The atmosphere that generated the cube is also saved for replicating results.  The cube is saved in an h5 file.

    Parameters
    ----------
    radiance_cube : 2D Array (float64)
        Spectral dimension first followed by spatial dimensions flattened.  This is the array returned from get_all_pixel_radiances

    atmos : List
        Contains all atmospheric objects such as wavenumbers, emissivities, P, T, H2O, O3, z, La, Ld, tau

    temp_param_list : List length 2 (float)
      temperature [K], followed by the temperature variance for normally distributed pixel temperatures
    
    labeled_data : 2D Array (int)
        Contains numeric class assignment for each pixel

    save_path : string
        Where the data cube will be saved

    Returns
    ----------
    """
    X, emis, P, T, H2O, O3, z, La, Ld, tau = atmos

    # Save as HDF5 file
    hf = h5py.File(save_path, 'w')
    d = hf.create_dataset('Radiance_Data', data=radiance_cube)
    d.attrs['surf_temp'] = temp_param_list[0]
    d.attrs['var_temp'] = temp_param_list[1]

    d = hf.create_dataset('Labels', data=labeled_data)

    # The rest of the parameters save the atmosphere that generated the cube
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

if __name__=='__main__':
    
    # Select one atmosphere to generate data cube, assumes no change in atmosphere across the scene
    temp_params = [287.0, 0.1] # Mean, variance

    truth_map = np.genfromtxt('data/pixel_predictions.csv', dtype=float, delimiter=",").astype(int)

    class_names = ['Scrub', 'Will_Smp', 'CP hammock', 'CP Oak', 'Slash Pine', 'Oak Broadleaf', 'Hardwood swamp',
                        'Graminoid Marsh', 'Spartina Marsh', 'Cattail Marsh', 'Salt Marsh', 'Mud Flats', 'Water']
    
    local_atmosphere = select_atmosphere(temp_params)

    L = get_all_pixel_radiances(local_atmosphere, truth_map, class_names, temp_params, all_pixels=True)

    data_cube_path = os.path.join('data', 'Ts_{}_cube.h5'.format(temp_params[0]))
    save_data_cube(L, local_atmosphere, temp_params, truth_map, data_cube_path)
    
    # The plot_augmented_radiance function was used primarily for debugging, but will be used in the future as part of a utility package for working with data cubes.  Currently, you need to generate a cube just to view these plots. 
    
    # plot_augmented_radiance(L, local_atmosphere, class_names, "Hardwood swamp")