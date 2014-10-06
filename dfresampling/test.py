import numpy as np
from dfresampling_cython import map_coordinates
import matplotlib.pyplot as plt
import sunpy.map
import sunpy.wcs

def scale(x, old_range, new_range):
    return (x - old_range[0]) / (old_range[1] - old_range[0]) * (new_range[1] - new_range[0]) + new_range[0]

source_map = sunpy.map.Map(sunpy.AIA_171_IMAGE)

def polar_to_cartesian(polar_pixel):
    plt.figure(figsize=(8,8))
    plt.subplot(221)
    plt.imshow(polar_pixel[:,:,0])
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(polar_pixel[:,:,1])
    plt.colorbar()

    original_shape = polar_pixel.shape
    polar_pixel = polar_pixel.reshape((polar_pixel.size/2, 2))
    polar = np.zeros_like(polar_pixel)
    cartesian = np.zeros_like(polar_pixel)
    cartesian_pixel = np.zeros_like(polar_pixel)
    
    polar[:,0] = scale(polar_pixel[:,0], (0, 2047), (0, 1200))
    polar[:,1] = scale(polar_pixel[:,1], (0, 2047), (-np.pi, np.pi))
    
    cartesian[:,0] = polar[:,0] * np.cos(polar[:,1])
    cartesian[:,1] = polar[:,0] * np.sin(polar[:,1])

    cartesian_pixel[:,0], cartesian_pixel[:,1] = sunpy.wcs.convert_data_to_pixel(cartesian[:,0], cartesian[:,1], scale=(source_map.scale["x"], source_map.scale["y"]), reference_pixel=(source_map.reference_pixel["x"], source_map.reference_pixel["y"]), reference_coordinate=(source_map.reference_coordinate["x"], source_map.reference_coordinate["y"]))

    cartesian_pixel = cartesian_pixel.reshape(original_shape)

    plt.subplot(223)
    plt.imshow(cartesian_pixel[:,:,0])
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(cartesian_pixel[:,:,1])
    plt.colorbar()
    plt.show()

    return cartesian_pixel

if __name__ == "__main__":
    source = source_map.data.astype(np.float64)
    target = np.zeros((2048, 2048))

    map_coordinates(source, target, polar_to_cartesian)

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.imshow(source)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(target)
    plt.colorbar()
    plt.show()


