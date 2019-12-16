import os
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from shapely import affinity
import geopandas as gpd
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.crs import CRS
from skimage import io, measure, transform
from skimage.morphology import dilation, disk


def set_env_vars(env_name):
    user_dir = os.path.expanduser('~')
    path_parts = ('AppData', 'Local', 
                  'Continuum', 'anaconda3')
    conda_dir = Path(user_dir).joinpath(*path_parts)
    env_dir = conda_dir / 'envs' / env_name
    share_dir = env_dir / 'Library' / 'share'
    script_path = conda_dir / 'Scripts'
    gdal_data_path = share_dir / 'gdal'
    proj_lib_path = share_dir

    if script_path.name not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + str(script_path)
    os.environ["GDAL_DATA"] = str(gdal_data_path)


if __name__ == '__main__':

    set_env_vars('shore_gen')

    img_s0_path = Path(r'Z:\ShoreGen\US4AK4LF_POLY.tif')
    img_s1_path = Path(r'Z:\ShoreGen\US4AK4LF_POLY_s1.tif')

    print('reading image...')
    img = io.imread(img_s0_path)
    img_rasterio = rasterio.open(img_s0_path)

    rescale_factor = 0.1
    
    epsg = 26905
    crs = CRS.from_epsg(epsg)

    simplify = 20
    smoothing = None

    n = 4250
    e = 16500
    s = 6000
    w = 14500

    n_r = n * rescale_factor
    e_r = e * rescale_factor
    s_r = s * rescale_factor
    w_r = w * rescale_factor

    img_s1_height = int(img.shape[0] * rescale_factor)
    img_s1_width = int(img.shape[1] * rescale_factor)
    output_shape = (1, img_s1_height, img_s1_width)
    print(output_shape)

    print('rescaling image ({})...'.format(rescale_factor))
    img_s1_ski = transform.resize(img, output_shape[1:], anti_aliasing=True)

    print('dilating skimage resized LNDARE...')
    disk_size = 1
    img_s1_ski = dilation(img_s1_ski, disk(disk_size))

    print('resampling rasterio img...')
    with rasterio.open(str(img_s0_path)) as f:
        img_s1_rio = f.read(out_shape=output_shape,
                            resampling=Resampling.average)

    profile = img_rasterio.profile
    t = img_rasterio.meta['transform']
    
    resize_offset = 1 / rescale_factor

    transform = Affine(t.a / rescale_factor, t.b, t.c + resize_offset, 
                       t.d, t.e / rescale_factor, t.f - resize_offset)

    shapely_affine = (t.a / rescale_factor, t.b, t.d,
                      t.e / rescale_factor, 
                      t.c + resize_offset, 
                      t.f - resize_offset)
    
    profile.update(
        height=img_s1_height,
        width=img_s1_width,
        count=1,
        crs=crs.wkt,
        transform=transform)

    print('writing {}...'.format(img_s1_path))
    with rasterio.open(str(img_s1_path), 'w', **profile) as dst:
        dst.write(img_s1_rio.squeeze(), 1)

    img_s1_rio = np.ma.array(img_s1_rio.squeeze(), mask=(img_s1_rio == 1))

    print('creating contours...')
    contours_SKI = measure.find_contours(img_s1_ski, 0.0)
    contours_RIO = measure.find_contours(img_s1_rio, 0.0)
        
    contours_list_SKI = []
    for c in contours_SKI:
        c[:, [0, 1]] = c[:, [1, 0]]
        c_trans = affinity.affine_transform(LineString(c), shapely_affine)
        contours_list_SKI.append(c_trans)

    contours_list_RIO = []
    for c in contours_RIO:
        c[:, [0, 1]] = c[:, [1, 0]]
        c_trans = affinity.affine_transform(LineString(c), shapely_affine)
        contours_list_RIO.append(c_trans)

    contour_gpkg = Path(r'Z:\ShoreGen\US4AK4LF_SL.gpkg')
    rescale_factor_int = int(1 / rescale_factor)

    gdf = gpd.GeoDataFrame(geometry=contours_list_SKI, crs=crs.to_dict())
    gdf.geometry = gdf.geometry.simplify(tolerance=simplify, preserve_topology=True)
    lyr = 'SKI_sf{}'.format(rescale_factor_int)
    gdf.to_file(str(contour_gpkg), layer=lyr, driver='GPKG')

    gdf = gpd.GeoDataFrame(geometry=contours_list_RIO, crs=crs.to_dict())
    lyr = 'RIO_sf{}'.format(rescale_factor_int)
    gdf.to_file(str(contour_gpkg), layer=lyr, driver='GPKG')
