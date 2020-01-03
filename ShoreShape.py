import os
import json
import geopandas as gpd
from pathlib import Path
import arcpy

from Schema import Schema


class ShorelineTile:

    def __init__(self, params):
        self.set_params(params)

    def set_params(self, params):
        for i, p in enumerate(params):
            self.__dict__[p.name] = p.value

    def export(self):
        out_path =  proj_dir / '{}_ATTRIBUTED.shp'.format(shp.stem)
        gdf.to_file(out_path, driver='ESRI Shapefile')

        

if __name__ == '__main__':

    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)

    schema_path = Path(r'.\shoreline_schema.json')
    schema = Schema(schema_path)
    #arcpy.AddMessage(schema)

    proj_dir = Path(r'\\ngs-s-rsd\Lidar_Contract00\TX1803\Imagery\ortho\shp')
    shp_dir = proj_dir / 'tiles'
    
    slt = ShorelineTile(arcpy.GetParameterInfo())
    shp_paths = [Path(shp) for shp in slt.shp_paths.exportToString().split(';')]
    num_shps = len(shp_paths)

    for i, shp in enumerate(shp_paths, 1):
        arcpy.AddMessage('{} ({} of {})...'.format(shp.name, i, num_shps))
        gdf = gpd.read_file(str(shp))

        if not gdf.empty:

            # apply tile-wide attributes

            # determine FIPS Alpha(s)

            # determine NOAA Region(s)

            # output attributed gdf
