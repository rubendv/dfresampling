import numpy as np
import sunpy.wcs
from functools import partial, reduce

def scale(x, old_range, new_range):
    return (x - old_range[0]) / (old_range[1] - old_range[0]) * (new_range[1] - new_range[0]) + new_range[0]

def convert_pixel_to_hpc(scale, reference_pixel, reference_coordinate, pixel):
    crpix = np.array(reference_pixel)
    cdelt = np.array(scale)
    crval = np.array(reference_coordinate)
    hpc = np.zeros_like(pixel)
    hpc[...,0] = (pixel[...,0] - (crpix[0] - 1)) * cdelt[0] + crval[0]
    hpc[...,1] = (pixel[...,1] - (crpix[1] - 1)) * cdelt[1] + crval[1]
    return hpc

def convert_hpc_to_pixel(scale, reference_pixel, reference_coordinate, hpc):
    crpix = np.array(reference_pixel)
    cdelt = np.array(scale)
    crval = np.array(reference_coordinate)
    pixel = np.zeros_like(hpc)
    pixel[...,0] = (hpc[...,0] - crval[0]) / cdelt[0] + crpix[0] - 1
    pixel[...,1] = (hpc[...,1] - crval[1]) / cdelt[1] + crpix[1] - 1
    return pixel

def convert_polar_to_cartesian(polar):
    cartesian = np.zeros_like(polar)
    cartesian[...,0] = polar[...,0] * np.cos(polar[...,1])
    cartesian[...,1] = polar[...,0] * np.sin(polar[...,1])
    return cartesian

def convert_range(old_range_x, new_range_x, old_range_y, new_range_y, coords):
    newcoords = np.zeros_like(coords)
    newcoords[...,0] = scale(coords[...,0], old_range_x, new_range_x)
    newcoords[...,1] = scale(coords[...,1], old_range_y, new_range_y)
    return newcoords

def convert_cartesian_to_polar(cartesian):
    polar = np.zeros_like(cartesian)
    polar[...,0] = np.hypot(cartesian[...,0], cartesian[...,1])
    polar[...,1] = np.arctan2(cartesian[...,1], cartesian[...,0])
    return polar

def convert_hg_to_hpc(b0_deg, l0_deg, dsun_meters, angle_units, occultation, hg):
    hpc = np.zeros_like(hg)
    hpc[...,0], hpc[...,1] = sunpy.wcs.convert_hg_hpc(hg[...,0], hg[...,1], b0_deg, l0_deg, dsun_meters, angle_units, occultation)
    return hpc

def convert_sin_hg_to_hg(sin_hg):
    hg = np.zeros_like(sin_hg)
    hg[...,0] = sin_hg[...,0]/np.cos(np.deg2rad(sin_hg[...,1]))
    hg[(hg < -180) | (hg > 180)] = np.nan
    hg[...,1] = sin_hg[...,1]
    
    return hg

def convert_hg_to_sin_hg(hg):
    sin_hg = np.zeros_like(hg)
    sin_hg[...,0] = hg[...,0]*np.cos(np.deg2rad(hg[...,1]))
    sin_hg[...,1] = hg[...,1]
    return sin_hg

def convert_hpc_pixel_to_sin_hg_pixel(hg_shape, longitude_range, latitude_range, scale, reference_pixel, reference_coordinate, b0_deg, l0_deg, dsun_meters, angle_units, angle, hpc_pixel):
    return compose(\
        partial(convert_range, longitude_range, (0, hg_shape[1]), latitude_range, (0, hg_shape[0])), \
        partial(convert_hg_to_sin_hg), \
        partial(convert_hpc_to_hg, b0_deg, l0_deg, dsun_meters, angle_units), \
        partial(rotate, angle), \
        partial(convert_pixel_to_hpc, scale, reference_pixel, reference_coordinate))(hpc_pixel)

def convert_hpc_to_hg(b0_deg, l0_deg, dsun_meters, angle_units, hpc):
    hg = np.zeros_like(hpc)
    hg[...,0], hg[...,1] = sunpy.wcs.convert_hpc_hg(hpc[...,0], hpc[...,1], b0_deg, l0_deg, dsun_meters, angle_units)
    return hg

def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions)

def convert_polar_pixel_to_hpc_pixel(polar_shape, max_radius, scale, reference_pixel, reference_coordinate, polar_pixel):
    return compose(partial(convert_hpc_to_pixel, scale, reference_pixel, reference_coordinate), convert_polar_to_cartesian, partial(convert_range, (0, polar_shape[1]), (0, max_radius), (0, polar_shape[0]), (-np.pi, np.pi)))(polar_pixel)

def convert_hpc_pixel_to_polar_pixel(polar_shape, max_radius, scale, reference_pixel, reference_coordinate, hpc_pixel):
    return compose(partial(convert_range, (0, max_radius), (0, polar_shape[1]), (-np.pi, np.pi), (0, polar_shape[0])), convert_cartesian_to_polar, partial(convert_pixel_to_hpc, scale, reference_pixel, reference_coordinate))(hpc_pixel)

def convert_hpc_pixel_to_hg_pixel(hg_shape, longitude_range, latitude_range, scale, reference_pixel, reference_coordinate, b0_deg, l0_deg, dsun_meters, angle_units, angle, hpc_pixel):
    return compose(\
            partial(convert_range, longitude_range, (0, hg_shape[1]), latitude_range, (0, hg_shape[0])), \
            partial(convert_hpc_to_hg, b0_deg, l0_deg, dsun_meters, angle_units), \
            partial(rotate, angle), \
            partial(convert_pixel_to_hpc, scale, reference_pixel, reference_coordinate))(hpc_pixel)

def convert_hg_pixel_to_hpc_pixel(hg_shape, longitude_range, latitude_range, scale, reference_pixel, reference_coordinate, b0_deg, l0_deg, dsun_meters, angle_units, occultation, angle, hg_pixel):
    return compose(\
            partial(convert_hpc_to_pixel, scale, reference_pixel, reference_coordinate), \
            partial(rotate, -angle), \
            partial(convert_hg_to_hpc, b0_deg, l0_deg, dsun_meters, angle_units, occultation), \
            partial(convert_range, (0, hg_shape[1]), longitude_range, (0, hg_shape[0]), latitude_range))(hg_pixel)

def rotate(angle_degrees, coords):
    angle_rad = np.deg2rad(angle_degrees)
    rotated = np.zeros_like(coords)
    rotated[...,0] = coords[...,0]*np.cos(angle_rad) - coords[...,1]*np.sin(angle_rad)
    rotated[...,1] = coords[...,0]*np.sin(angle_rad) + coords[...,1]*np.cos(angle_rad)
    return rotated

def convert_sin_hg_pixel_to_hpc_pixel(hg_shape, longitude_range, latitude_range, scale, reference_pixel, reference_coordinate, b0_deg, l0_deg, dsun_meters, angle_units, occultation, angle, hg_pixel):
    return compose(\
            partial(convert_hpc_to_pixel, scale, reference_pixel, reference_coordinate), \
            partial(rotate, -angle), \
            partial(convert_hg_to_hpc, b0_deg, l0_deg, dsun_meters, angle_units, occultation), \
            convert_sin_hg_to_hg, \
            partial(convert_range, (0, hg_shape[1]), longitude_range, (0, hg_shape[0]), latitude_range))(hg_pixel)
