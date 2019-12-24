import os
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from shapely import affinity
import geopandas as gpd
import rasterio
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


class ImageTile:

    def __init__(self):
        self.tifs = []
        self.tifs_rgb = []
        self.out_meta = None

    def get_bands(self):
        pass

    def calc_index(self, dem, water_dir):
        img = io.imread(dem)
            

        #red =  img[:, :, 0]
        #green = img[:, :, 1]
        #blue = img[:, :, 2]
        #nir = img[:, :, 3]
            
        #ndwi = (green - nir) / (green + nir)
        #ndwi = np.where(np.isfinite(ndwi), ndwi, -1)
        #ndwi = np.expand_dims(ndwi, axis=2)

        #ndvi = (red - nir) / (red + nir)
        #ndvi = np.where(np.isfinite(ndvi), ndvi, -1)
        #ndvi = np.expand_dims(ndvi, axis=2)

        return index_raster

    def resize_boolean_raster(self, img):
        rescale_factor = 0.25
        img_s1_height = int(img.shape[0] * rescale_factor)
        img_s1_width = int(img.shape[1] * rescale_factor)
        output_shape = (1, img_s1_height, img_s1_width)
        print(output_shape)
        img_s1_ski = transform.resize(img, output_shape[1:], anti_aliasing=True)

        src = rasterio.open(dem)
            
        red =  img_s1_ski[:,:,0]
        green = img_s1_ski[:,:,1]
        blue = img_s1_ski[:,:,2]
        nir = img_s1_ski[:,:,3]
            
        ndwi = (green - nir) / (green + nir)
        ndwi = np.where(np.isfinite(ndwi), ndwi, -1)            
        water = np.where(ndwi > 0, 0, 1).astype(np.uint8)

        meta = src.profile
        t = meta['transform']
        resize_offset = 1 / rescale_factor
        t_affine = Affine(t.a / rescale_factor, t.b, t.c, 
                            t.d, t.e / rescale_factor, t.f)

        meta.update(count=1, 
                    height=x, 
                    width=y, 
                    #dtype='float64',
                    nodata=1,
                    transform=t_affine)

        ndwi_path = water_dir / '{}_Kmeans.tif'.format(dem.stem)
        with rasterio.open(ndwi_path, 'w', n_bits=1, **meta) as dst:
            dst.write(water, 1)

    def classify_tif(self):
        x, y, z = img_s1_ski.shape
        img_2d = img_s1_ski.reshape(x*y, z)

        print('K-means clustering...')
        k_cluster = cluster.KMeans(n_clusters=5, n_jobs=-6)
        k_cluster.fit(img_2d)
        k_centers = k_cluster.cluster_centers_
        k_labels = k_cluster.labels_

        cluster_colors = np.asarray([1, 2, 3, 4, 5])

        k_means_img = cluster_colors[k_labels].reshape(x, y).astype('uint8')

        plt.figure(figsize=(8, 8))
        plt.imshow(k_means_img)
        plt.show()

    def threshold_index_raster(self):
        pass

    def generalize():
        img_s0_path = Path(r'C:\QAQC_contract\WATER.tif')
        img_s1_path = Path(r'C:\QAQC_contract\WATER_s1.tif')

        print('reading image...')
        img = io.imread(img_s0_path)
        img_rasterio = rasterio.open(img_s0_path)

        rescale_factor = 0.1
    
        epsg = 26905  # NAD83/UTM Zone 5N
        epsg = 26914  # NAD83/UTM Zone 14N
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

        output_shape_SKI = (img_s1_height, img_s1_width, 4)
        output_shape_RAS = (1, img_s1_height, img_s1_width)
        print('output shape (Skimage):  {}'.format(output_shape_SKI))
        print('output shape (Rasterio):  {}'.format(output_shape_RAS))

        print('rescaling image ({})...'.format(rescale_factor))
        img_s1_ski = transform.resize(img, output_shape_SKI, anti_aliasing=True).astype('uint8')

        print('dilating skimage resized LNDARE...')
        disk_size = 1
        #img_s1_ski = dilation(img_s1_ski, disk(disk_size))

        print('resampling rasterio img...')
        with rasterio.open(str(img_s0_path)) as f:
            img_s1_rio = f.read(out_shape=output_shape_RAS,
                                resampling=Resampling.average)

        profile = img_rasterio.profile
        t = img_rasterio.meta['transform']
    
        resize_offset = 1 / rescale_factor

        t_affine = Affine(t.a / rescale_factor, t.b, t.c, 
                          t.d, t.e / rescale_factor, t.f)

        shapely_affine = (t.a / rescale_factor, t.b, t.d,
                          t.e / rescale_factor, t.c, t.f)
    
        profile.update(
            height=img_s1_height,
            width=img_s1_width,
            count=1,
            crs=crs.wkt,
            dtype='uint8',
            transform=t_affine)

        print('writing {}...'.format(img_s1_path))
        with rasterio.open(str(img_s1_path), 'w', **profile) as dst:
            dst.write(img_s1_ski, 4)

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

        contour_gpkg = Path(r'C:\QAQC_contract\WATER.gpkg')
        rescale_factor_int = int(1 / rescale_factor)

        gdf = gpd.GeoDataFrame(geometry=contours_list_SKI, crs=crs.to_dict())
        gdf.geometry = gdf.geometry.simplify(tolerance=simplify, preserve_topology=True)
        lyr = 'SKI_sf{}'.format(rescale_factor_int)
        gdf.to_file(str(contour_gpkg), layer=lyr, driver='GPKG')

        gdf = gpd.GeoDataFrame(geometry=contours_list_RIO, crs=crs.to_dict())
        lyr = 'RIO_sf{}'.format(rescale_factor_int)
        gdf.to_file(str(contour_gpkg), layer=lyr, driver='GPKG')

    def get_classified_dems(self, dem_dir):
        print('retreiving individual QL DEMs...')
        for dem in list(dem_dir.glob('*_Kmeans.tif')):
            src = rasterio.open(dem)
            self.tifs.append(src)        
            self.out_meta = src.meta.copy()  # uses last src made

    def mosaic(self, water_dir):
        water_path = water_dir / 'WATER.tif'
        self.get_classified_dems(water_dir)
        if self.tifs:
            print('generating {}...'.format(water_path))
            mosaic, out_trans = merge.merge(self.tifs)
            self.out_meta.update({
                'driver': "GTiff",
                'height': mosaic.shape[1],
                'width': mosaic.shape[2],
                'transform': out_trans})
            with rasterio.open(water_path, 'w', **self.out_meta) as dest:
                dest.write(mosaic)
        else:
            print('No DEM tiles were generated.')


class ShorelineRaster:

    def __init__(self):
        pass


class Shoreline:

    def __init__(self):
        pass

    def make_pretty():
        pass

def main():

    tif_dir = Path(r'\\ngs-s-rsd\Lidar_Contract00\TX1803\Imagery\ortho\UTM14')
    water_dir = Path(r'C:\QAQC_contract') 
    tif_paths = list(tif_dir.glob('*.tif'))
    num_tifs = len(list(tif_paths))
    water_path = water_dir / 'ShorelineMOSAIC.tif'

    img_tiles = []

    for i, dem in enumerate(tif_paths, 1):

        print('processing shoreline image tile {} if {}...'.format(i, num_tifs))
        img_tile = ImageTile(dem)
        img_tiles.append(img_tile)
        
        # get  tif bands
        bands = img_tile.get_bands(dem)

        # calc index raster
        index_raster = img_tile.calc_index(dem)

        # threshold index raster
        img_tile.threshold_index_raster(index_raster)

        # resize boolean raster
        img_tile.resize_boolean_raster()

    # mosaic shoreline boolean rasters
    shoreline_raster = ShorelineRaster()
    shoreline_raster.mosaic(img_tiles)
    
    # generalize boolean raster
    shoreline_raster.generalize()

    # extract vector shoreline
    shoreline = Shoreline(shoreline_raster.to_vector())

    # generalized contours
    shoreline.make_pretty()


if __name__ == '__main__':
    set_env_vars('shore_gen')
    main()
