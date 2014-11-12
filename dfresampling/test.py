import numpy as np
from dfresampling import map_coordinates, map_coordinates_direct
from dfresampling.projections import partial, convert_hpc_pixel_to_polar_pixel, convert_polar_pixel_to_hpc_pixel, convert_sin_hg_pixel_to_hpc_pixel
import matplotlib.pyplot as plt
import sunpy.map
import sunpy.wcs
import sunpy.cm
from matplotlib import colors
import matplotlib.cm
from scipy.ndimage import uniform_filter, gaussian_filter1d
import sys
import skimage.io

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
    print source_map.date
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

    plt.figure(figsize=(8,8))
    vmin = 0
    vmax = np.percentile(source, 99.9)
    ax1 = plt.subplot(121)
    ax1.set_title("Original")
    ax1.imshow(source, vmin=vmin, vmax=vmax, cmap=sunpy.cm.get_cmap("sdoaia171"), interpolation="nearest")

    sin_hg = np.zeros((1024, 1024))
    map_coordinates(source, sin_hg, partial(convert_sin_hg_pixel_to_hpc_pixel, sin_hg.shape, (-90, 90), (-90, 90), scale, reference_pixel, reference_coordinate, source_map.heliographic_latitude, 0.0, source_map.dsun, "arcsec", True), progress=True) 

    ax2 = plt.subplot(122)
    ax2.set_title("Sinusoidal projection")
    ax2.imshow(sin_hg, vmin=vmin, vmax=vmax, cmap=sunpy.cm.get_cmap("sdoaia171"), interpolation="nearest")
    plt.show()

    skimage.io.imsave("test.png", matplotlib.cm.ScalarMappable(norm=colors.Normalize(vmin, vmax), cmap=sunpy.cm.get_cmap("sdoaia171")).to_rgba(sin_hg))
