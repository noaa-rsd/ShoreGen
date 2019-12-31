import os
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from shapely import affinity
import geopandas as gpd
import rasterio as rio
from rasterio import Affine, merge, plot
from rasterio.enums import Resampling
from rasterio.crs import CRS
from skimage import io, measure, transform
from skimage.morphology import dilation, disk
from sklearn import cluster
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors


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


class ShorelineRasterTile:

    def __init__(self, srt_path):
        self.srt_path = srt_path
        self.img = io.imread(srt_path)
        self.profile = self.set_profile()
        self.rescale_factor = 0.25
        self.red = self.img[:, :, 0]
        self.green = self.img[:, :, 1]
        self.blue = self.img[:, :, 2]
        self.nir = self.img[:, :, 3]

    def set_profile(self):
        src = rio.open(self.srt_path)
        return src.profile

    def get_profile(self, resized_shape, dtype):
        profile = self.profile.copy()
        profile.update({
            'count': 1,
            'height': resized_shape[0], 
            'width': resized_shape[1], 
            'dtype': dtype,
            'transform': self.get_resized_transform()})
        return profile

    def get_resized_transform(self):
        t = self.profile['transform']
        return  Affine(t.a / self.rescale_factor, t.b, t.c,
                       t.d, t.e / self.rescale_factor, t.f)

    def calc_ndvi(self):
        np.seterr(divide='ignore', invalid='ignore')
        ndvi_numerator = self.nir.astype(float) - self.red.astype(float)
        ndvi_demoninator = self.nir + self.red
        ndvi = ndvi_numerator / ndvi_demoninator
        ndvi = np.where(np.isfinite(ndvi), ndvi, 0)
        return ndvi

    def resize(self, srt):
        rescaled_height = int(srt.shape[0] * self.rescale_factor)
        rescaled_width = int(srt.shape[1] * self.rescale_factor)
        rescaled_shape = (rescaled_height, rescaled_width)
        resized_srt = transform.resize(srt, rescaled_shape, 
                                       anti_aliasing=True) 
        return resized_srt, rescaled_shape

    def export(self, resized_srt, resized_shape, dtype, export_path):
        profile = self.get_profile(resized_shape, dtype)
        with rio.open(export_path, 'w', **profile) as dst:
            dst.write(resized_srt, 1)

    def k_means(self, img, num_kmeans):
        x, y, z = img.shape
        img_2d = img.reshape(x*y, z)

        print('K-means clustering...')
        k_cluster = cluster.KMeans(n_clusters=num_kmeans, n_jobs=-6)
        k_cluster.fit(img_2d)
        k_centers = k_cluster.cluster_centers_
        k_labels = k_cluster.labels_

        cluster_colors = np.arange(num_kmeans)
        k_means_img = cluster_colors[k_labels].reshape(x, y)

        plt.figure(figsize=(8, 8))
        plt.imshow(k_means_img)
        plt.show()

        return k_means_img


class ShorelineRaster:

    def __init__(self, img_dir):
        self.srt_sources = self.get_shoreline_raster_tiles(img_dir)
        self.out_profile = self.srt_sources[-1].profile.copy()
        self.mosaic_trans = None

    def get_shoreline_raster_tiles(self, img_dir):
        shoreline_raster_tiles = []
        print('retreiving shoreline raster tiles...')
        for i, srt in enumerate(list(img_dir.glob('*_SRT.tif'))):
            src = rio.open(srt)
            shoreline_raster_tiles.append(src)        
        return shoreline_raster_tiles

    def update_profile(self, dilated_mosaic):
        self.out_profile.update({
            'driver': "GTiff",
            'height': dilated_mosaic.shape[0],
            'width': dilated_mosaic.shape[1],
            'transform': self.mosaic_transform})

    def mosaic(self):
        mosaic, self.mosaic_transform = merge.merge(self.srt_sources)
        return mosaic
    
    def dilate(self, mosaic, export_path, disk_size=1):
        print('dilating shoreline raster...')
        dilated_mosaic = dilation(mosaic.squeeze(), disk(disk_size))
        if export_path:
            self.update_profile(dilated_mosaic)
            with rio.open(str(export_path), 'w', **self.out_profile) as dst:
                dst.write(dilated_mosaic, 1)
        return dilated_mosaic

    def to_vector(self, dilated_mosaic):
        contours = measure.find_contours(dilated_mosaic, 0.0)
        t = self.mosaic_transform
        shapely_affine = (t.a, t.b, t.d, t.e, t.c, t.f)
        contours_list = []
        for c in contours:
            c[:, [0, 1]] = c[:, [1, 0]]
            c_trans = affinity.affine_transform(LineString(c), shapely_affine)
            contours_list.append(c_trans)
        return contours_list


class Shoreline:

    def __init__(self, contours_list, epsg):
        crs = CRS.from_epsg(epsg)
        self.gdf = gpd.GeoDataFrame(geometry=contours_list, crs=crs.to_dict())

    def make_pretty(self, simplify=20, smoothing=None):
        self.gdf.geometry = self.gdf.geometry.simplify(tolerance=simplify, 
                                                       preserve_topology=True)

    def export(self, gpkg, lyr):
        self.gdf.to_file(gpkg, layer=lyr, driver='GPKG')


def main():

    tif_dir = Path(r'\\ngs-s-rsd\Lidar_Contract00\TX1803\Imagery\ortho\UTM14')
    tif_paths = list(tif_dir.glob('*.tif'))[3:6]
    num_tifs = len(tif_paths)
    out_dir = Path(r'C:\QAQC_contract') 

    epsgs = {
        'NAD83/UTM Zone 5N': 26905,
        'NAD83/UTM Zone 14N': 26914,
        }
    crs_epsg = epsgs['NAD83/UTM Zone 14N']

    for i, tif_path in enumerate(tif_paths, 1):

        print('processing shoreline image tile {} if {}...'.format(i, num_tifs))
        srt = ShorelineRasterTile(tif_path)

        ## calc index raster
        ndvi_raster = srt.calc_ndvi()

        # calc kmean clusters
        resized_img, resized_shape = srt.resize(srt.img)
        num_kmeans = 6
        kmeans_raster = srt.k_means(resized_img, num_kmeans)
        export_path = out_dir / '{}_{}kmeans.tif'.format(tif_path.stem, num_kmeans)
        dtype = 'uint8'
        srt.export(kmeans_raster.astype(dtype), resized_shape, dtype, export_path)

        # resize boolean raster
        resized_srt, resized_shape = srt.resize(ndvi_raster)

        # export resized boolean shoreline raster (SR)
        export_path = out_dir / '{}_SRT.tif'.format(tif_path.stem)
        srt.export(resized_srt, resized_shape, 'float64', export_path)

    # mosaic shoreline boolean rasters
    shoreline_raster = ShorelineRaster(out_dir)
    mosaic = shoreline_raster.mosaic()
    
    # generalize boolean raster
    export_path = out_dir / 'ShorelineRaster.tif'
    dilated_mosaic = shoreline_raster.dilate(mosaic, export_path)

    # extract vector shoreline
    contour_geomety = shoreline_raster.to_vector(dilated_mosaic)
    shoreline = Shoreline(contour_geomety, crs_epsg)

    # carogrpahically generalize contours
    shoreline.make_pretty()

    # export cartographically generalized shoreline
    contour_gpkg = out_dir / 'Shoreline.gpkg'
    lyr = 'NDVI'
    shoreline.export(str(contour_gpkg), lyr)


if __name__ == '__main__':
    set_env_vars('shore_gen')
    main()
