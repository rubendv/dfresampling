import numpy as np
from dfresampling_cython import map_coordinates
import matplotlib.pyplot as plt
import sunpy.map
import sunpy.wcs
import sunpy.cm
from matplotlib import colors
from scipy.ndimage import uniform_filter

source_map = sunpy.map.Map(sunpy.AIA_171_IMAGE)
polar_shape = ((2048, 1024))
max_radius = 1200

def scale(x, old_range, new_range):
    return (x - old_range[0]) / (old_range[1] - old_range[0]) * (new_range[1] - new_range[0]) + new_range[0]

def convert_pixel_to_data(x, y, scale, reference_pixel, reference_coordinate):
    crpix = np.array(reference_pixel)
    cdelt = np.array(scale)
    crval = np.array(reference_coordinate)
    coordx = (x - (crpix[0] - 1)) * cdelt[0] + crval[0]
    coordy = (y - (crpix[1] - 1)) * cdelt[1] + crval[1]
    return coordx, coordy

def polar_to_cartesian(polar_pixel):
    original_shape = polar_pixel.shape
    polar_pixel = polar_pixel.reshape((polar_pixel.size/2, 2))
    polar = np.zeros_like(polar_pixel)
    cartesian = np.zeros_like(polar_pixel)
    cartesian_pixel = np.zeros_like(polar_pixel)
    
    polar[:,0] = scale(polar_pixel[:,0], (0, polar_shape[1]-1), (0, max_radius))
    polar[:,1] = scale(polar_pixel[:,1], (0, polar_shape[0]-1), (-np.pi, np.pi))
    
    cartesian[:,0] = polar[:,0] * np.cos(polar[:,1])
    cartesian[:,1] = polar[:,0] * np.sin(polar[:,1])

    cartesian_pixel[:,0], cartesian_pixel[:,1] = sunpy.wcs.convert_data_to_pixel(cartesian[:,0], cartesian[:,1], scale=(source_map.scale["x"], source_map.scale["y"]), reference_pixel=(source_map.reference_pixel["x"], source_map.reference_pixel["y"]), reference_coordinate=(source_map.reference_coordinate["x"], source_map.reference_coordinate["y"]))

    return cartesian_pixel.reshape(original_shape)

def cartesian_to_polar(cartesian_pixel):
    original_shape = cartesian_pixel.shape
    cartesian_pixel = cartesian_pixel.reshape((cartesian_pixel.size/2, 2))
    polar = np.zeros_like(cartesian_pixel)
    cartesian = np.zeros_like(cartesian_pixel)
    polar_pixel = np.zeros_like(cartesian_pixel)
    
    cartesian[:,0], cartesian[:,1] = convert_pixel_to_data(cartesian_pixel[:,0], cartesian_pixel[:,1], scale=(source_map.scale["x"], source_map.scale["y"]), reference_pixel=(source_map.reference_pixel["x"], source_map.reference_pixel["y"]), reference_coordinate=(source_map.reference_coordinate["x"], source_map.reference_coordinate["y"]))
    
    polar[:,0] = np.hypot(cartesian[:,0], cartesian[:,1])
    polar[:,1] = np.arctan2(cartesian[:,1], cartesian[:,0])

    polar_pixel[:,0] = scale(polar[:,0], (0, max_radius), (0, polar_shape[1]-1))
    polar_pixel[:,1] = scale(polar[:,1], (-np.pi, np.pi), (0, polar_shape[0]-1))

    return polar_pixel.reshape(original_shape)

def scale4_forward(input_pixel):
    return 4*input_pixel

def scale4_backward(input_pixel):
    return 0.25*input_pixel

def normalizer(image):
    image[np.isnan(image)] = 0
    mean = np.mean(image)
    std = np.std(image)

    vmin = max(0, mean - 3 * std)
    vmax = min(np.max(image), mean + 3 * std)

    return colors.Normalize(vmin, vmax)

if __name__ == "__main__":
    source = source_map.data.astype(np.float64)
    polar = np.zeros(polar_shape)
    cartesian = np.zeros((4096, 4096))

    print "Projecting to polar..."
    map_coordinates(source, polar, polar_to_cartesian)
    polar[np.isnan(polar)] = 0
    profile = np.median(polar, axis=0)
    plt.figure()
    plt.title("Radial profile")
    plt.plot(profile)
    profile = profile.reshape((1, profile.size))
    print "Projecting to cartesian..."
    map_coordinates(polar, cartesian, lambda x: cartesian_to_polar(scale4_backward(x)), max_samples_width=128)
    cartesian[np.isnan(cartesian)] = 0

    cartesian = uniform_filter(cartesian, size=4)[::4,::4]*16

    hpc_normalizer = normalizer(source)
    polar_normalizer = normalizer(polar)

    plt.figure(figsize=(8,4))
    ax1 = plt.subplot(131)
    plt.title("Original")
    plt.imshow(hpc_normalizer(source), vmin=0, vmax=1, interpolation="nearest", cmap=sunpy.cm.get_cmap("sdoaia171"))
    ax2 = plt.subplot(132)
    plt.title("Polar projection")
    plt.imshow(polar_normalizer(polar), vmin=0, vmax=1, interpolation="nearest", cmap=sunpy.cm.get_cmap("sdoaia171"))
    ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
    plt.title("Cartesian projection of polar image")
    plt.imshow(hpc_normalizer(cartesian), vmin=0, vmax=1, interpolation="nearest", cmap=sunpy.cm.get_cmap("sdoaia171"))
    plt.show()


