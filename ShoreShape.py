import json
import geopandas as gpd
from pathlib import Path

from Schema import Schema


class Shoreline:

    def __init__(self):
        pass


if __name__ == '__main__':

    schema_path = Path(r'Z:\ShoreGen\ShoreGen_noaa-rsd\shoreline_schema.json')
    schema = Schema(schema_path)
    print(schema)

    #proj_dir = Path(r'\\ngs-s-rsd\Lidar_Contract00\TX1803\Imagery\ortho\shp')
    #shp_dir = proj_dir / 'tiles'

    #shps = list(shp_dir.glob('*.shp'))
    #num_shps = len(shps)
    #for i, shp in enumerate(shps, 1):
    #    print('processing {} ({} of {})...'.format(shp.name, i, num_shps))
    #    gdf = gpd.read_file(str(shp))

    #    #if not gdf.empty:
    #    #    gpkg_path =  proj_dir / 'TX1803.gpkg'
    #    #    layer = 'shp_{}'.format(shp.stem)
    #    #    gdf.to_file(gpkg_path, layer=layer, driver='GPKG')
