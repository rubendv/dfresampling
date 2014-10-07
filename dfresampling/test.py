import numpy as np
from dfresampling import map_coordinates
from dfresampling.projections import partial, convert_hpc_pixel_to_polar_pixel, convert_polar_pixel_to_hpc_pixel
import matplotlib.pyplot as plt
import sunpy.map
import sunpy.wcs
import sunpy.cm
from matplotlib import colors
from scipy.ndimage import uniform_filter, gaussian_filter1d
import sys

source_map = sunpy.map.Map(sunpy.AIA_171_IMAGE)
polar_shape = ((1024, 512))
max_radius = source_map.rsun_arcseconds * 1.2


def normalizer(image):
    image[np.isnan(image)] = 0
    mean = np.mean(image)
    std = np.std(image)

    vmin = max(0, mean - 3 * std)
    vmax = min(np.max(image), mean + 3 * std)

    return colors.Normalize(vmin, vmax)

if __name__ == "__main__":
    source = source_map.data.astype(np.float64) / source_map.exposure_time
    polar = np.zeros(polar_shape)
    cartesian = np.zeros((4096, 4096))

    scale = (source_map.scale["x"], source_map.scale["y"])
    reference_pixel = (source_map.reference_pixel["x"], source_map.reference_pixel["y"])
    reference_coordinate = (source_map.reference_coordinate["x"], source_map.reference_coordinate["y"])

    print "Projecting to polar..."
    map_coordinates(source, polar, partial(convert_polar_pixel_to_hpc_pixel, polar_shape, max_radius, scale, reference_pixel, reference_coordinate))
    polar[np.isnan(polar)] = 0
    profile = np.mean(polar, axis=0)
    plt.figure()
    plt.title("Radial profile")
    x = np.linspace(0, max_radius/source_map.rsun_arcseconds, polar.shape[1])
    ax = plt.subplot(111)
    median = np.median(polar, axis=0)
    plt.plot(x, np.mean(polar, axis=0), label="Mean")
    plt.plot(x, median, label="Median")
    plt.plot(x, np.where(x < 0.96, gaussian_filter1d(median, sigma=8), median), label="Smoothed median")
    plt.legend(loc="best")
    plt.show()
    sys.exit()
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


