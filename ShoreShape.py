import os
import json
import geopandas as gpd
from pathlib import Path
import arcpy
from datetime import datetime

from Schema import Schema


class ShorelineTile:

    def __init__(self, params):
        self.set_params(params)
        self.gdf = None

    def set_params(self, params):
        for i, p in enumerate(params):
            self.__dict__[p.name] = p.value

    def populate_gdf(self, shp):
        self.gdf = gpd.read_file(str(shp))

    def export(self, out_path):
        self.gdf.to_file(out_path, driver='ESRI Shapefile')


if __name__ == '__main__':

    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)

    schema_path = Path(r'.\shoreline_schema.json')
    schema = Schema(schema_path)

    proj_dir = Path(r'\\ngs-s-rsd\Lidar_Contract00\TX1803\Imagery\ortho\shp')
    shp_dir = proj_dir / 'tiles'
    
    noaa_region_states_path = Path(r'Z:\ShoreGen\ShoreGen_noaa-rsd\support\shoreline_state_regions.shp')
    noaa_region_states_gdf = gpd.read_file(str(noaa_region_states_path))

    slt = ShorelineTile(arcpy.GetParameterInfo())
    shp_paths = [Path(shp) for shp in slt.shp_paths.exportToString().split(';')]
    num_shps = len(shp_paths)

    for i, shp in enumerate(shp_paths, 1):

        arcpy.AddMessage('{} ({} of {})...'.format(shp.name, i, num_shps))
        slt.populate_gdf(shp)

        if not slt.gdf.empty:

            # apply tile-wide attributes
            for attribute in schema.atypes['tile']:
                slt.gdf[attribute] = slt.__dict__[attribute]
                
                if attribute == 'SRC_DATE':
                    slt.gdf[attribute] = slt.gdf[attribute].apply(lambda x: datetime.strftime(x, '%Y%m'))

            # determine FIPS Alpha(s)
            arcpy.AddMessage(noaa_region_states_gdf.geometry.contains(slt.gdf.geometry))

            # determine NOAA Region(s)

            # output attributed gdf
            out_path =  proj_dir / '{}_ATTRIBUTED.shp'.format(shp.stem)
            slt.export(out_path)
