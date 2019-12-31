import geopandas as gpd
from pathlib import Path

shp_path = Path(r'\\ngs-s-rsd\Lidar_Contract00\TX1803\Imagery\ortho\shp\TX1803-UTM14-NDWI.shp')

print('reading {}...'.format(shp_path))
gdf = gpd.read_file(str(shp_path))
print(gdf)

gpkg_path = Path(r'\\ngs-s-rsd\Lidar_Contract00\TX1803\Imagery\ortho\shp\TX1803.gpkg')
layer = 'NDWI_Contour'
print('writing gpkg...')
gdf.to_file(gpkg_path, layer=layer, driver='GPKG')


