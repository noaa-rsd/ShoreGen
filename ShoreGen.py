import skimage.io
import skimage.measure
import skimage.transform
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely import affinity
#import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio import Affine
from rasterio.crs import CRS


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

    img_path = Path(r'Z:\ShoreGen\US4AK4LF_POLY.tif')
    img_s1_path = Path(r'Z:\ShoreGen\US4AK4LF_POLY_s1.tif')

    print('reading image...')
    img = skimage.io.imread(img_path)
    img_rasterio = rasterio.open(img_path)

    rescale_factor = 0.1
    
    n = 4250
    e = 16500
    s = 6000
    w = 14500

    n_r = n / rescale_factor
    e_r = e / rescale_factor
    s_r = s / rescale_factor
    w_r = w / rescale_factor

    img_s1_height = int(img.shape[0] * rescale_factor)
    img_s1_width = int(img.shape[1] * rescale_factor)
    output_shape = (1, img_s1_height, img_s1_width)

    print('rescaling image ({})...'.format(rescale_factor))
    img_s1_ski = skimage.transform.resize(img, output_shape[1:], 
                                          anti_aliasing=True)

    with rasterio.open(str(img_path)) as f:
        img_s1_rio = f.read(out_shape=output_shape,
                            resampling=Resampling.average)

    profile = img_rasterio.profile
    t = img_rasterio.meta['transform']
    
    transform = Affine(t.a / rescale_factor, t.b, t.c, 
                       t.d, t.e / rescale_factor, t.f)
    gdal_affine_matrix = transform.to_gdal()
    
    epsg = 26905
    crs = CRS.from_epsg(epsg)

    profile.update(
        height=img_s1_height,
        width=img_s1_width,
        count=1,
        crs=crs.wkt,
        transform=transform)

    with rasterio.open(str(img_s1_path), 'w', **profile) as dst:
        dst.write(img_s1_rio.squeeze(), 1)

    img_s1_rio = np.ma.array(img_s1_rio.squeeze(), mask=(img_s1_rio == 1))

    print('creating contours...')
    contours_SKI = skimage.measure.find_contours(img_s1_ski, 0.0)
    contours_RIO = skimage.measure.find_contours(img_s1_rio, 0.0)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(img_s1_ski, cmap='gray', alpha=0.5)
    ax[1].set_title("img_resized_ski")

    ax[1].imshow(img_s1_rio, cmap='Blues', alpha=0.5)
    ax[1].set_title("img_resized_ski")

    ax[0].set_xlim(w, e)
    ax[0].set_ylim(s, n)
    
    ax[1].set_xlim(w_r, e_r)
    ax[1].set_ylim(s_r, n_r)
    
    contour_list = []
    for i, c in enumerate(contours_SKI):
        ax[1].plot(c[:, 1], c[:, 0], linewidth=1, color='orange')
        l_orig = LineString(c)
        l_trans = affinity.affine_transform(l_orig, gdal_affine_matrix)
        contour_list.append(l_trans)
        print(contour_list[-1])

    contour_list = []
    for i, c in enumerate(contours_RIO):
        ax[1].plot(c[:, 1], c[:, 0], linewidth=1, color='blue')
        contour_list.append(LineString(c))

    #wgs84 = {'init': 'epsg:4326'}
    #gdf = gpd.GeoDataFrame(geometry=contour_list, crs=wgs84)
    #print(gdf)

    #contour_gpkg = Path(r'Z:\ShoreGen\US4AK4LF_SL.geojson')
    #gdf.to_file(str(contour_gpkg), driver='GeoJSON')

    plt.tight_layout()
    plt.show()
