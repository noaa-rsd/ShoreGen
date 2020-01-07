import os
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
import arcpy
from datetime import datetime
from shapely.geometry import Polygon, MultiLineString
from matplotlib import pyplot as plt

from Schema import Schema


class ShorelineTile():

    def __init__(self, params, schema):
        self.set_params(params)
        self.schema = schema
        self.gdf = None

    def set_params(self, params):
        for i, p in enumerate(params):
            self.__dict__[p.name] = p.value

    def populate_gdf(self, shp):
        self.gdf = gpd.read_file(str(shp))

    def export(self, out_path):
        self.gdf.to_file(out_path, driver='ESRI Shapefile')

    def apply_tile_attributes(self):
        for attr in self.schema.atypes['tile']:
            if attr != 'SRC_DATE':
                self.gdf[attr] = str(self.__dict__[attr])
            elif attr == 'SRC_DATE':  # dytpe is datetime64 (can't be)
                self.gdf[attr] = datetime.strftime(self.__dict__[attr], '%Y%m')

    def apply_state_region_attributes(self, state_regions):
        if state_regions.shape[0] == 1:
            self.gdf['ATTRIBUTE'] = None
            self.gdf['FIPS_ALPHA'] = state_regions.iloc[0]['STATE_FIPS']
            self.gdf['NOAA_Regio'] = state_regions.iloc[0]['NOAA_Regio']
        elif state_regions.shape[0] > 1:
            shoreline_gdfs = []
            for i, state_region in state_regions.iterrows():
                arcpy.AddMessage('get shoreline intersecting state_region...')
                sindex = self.gdf.sindex
                intersect_idx = list(sindex.intersection(state_region.geometry.bounds))
                possible_shoreline = self.gdf.iloc[intersect_idx]

                arcpy.AddMessage('clip intersecting shoreline with region...')  # TODO: TAKING LONG TIME
                arcpy.AddMessage(MultiLineString(list(possible_shoreline.geometry)))
                shoreline = possible_shoreline.intersection(state_region.geometry)  # TODO: need GeoSeries or GeoDataFrame?
                shoreline = gpd.GeoDataFrame(geometry=shoreline, crs=self.gdf.crs)

                arcpy.AddMessage('attribute state_region shoreline...')
                shoreline['ATTRIBUTE'] = None
                shoreline['FIPS_ALPHA'] = state_region['STATE_FIPS']
                shoreline['NOAA_Regio'] = state_region['NOAA_Regio']

                shoreline_gdfs.append(shoreline)

            df = pd.concat(shoreline_gdfs, ignore_index=True)
            self.gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=self.gdf.crs)

    def get_overlapping_state_regions(self):
        #noaa_region_states_path = Path(r'.\support\state_regions.shp')
        noaa_region_states_path = Path(r'.\support\state_regions_FAUX_TEST.shp')
        state_regions = gpd.read_file(str(noaa_region_states_path))
        state_regions = state_regions.to_crs(self.gdf.crs)
        sindex = state_regions.sindex
        extents = self.get_tile_extents()
        region_states_idx = list(sindex.intersection(extents.bounds))
        return state_regions.iloc[region_states_idx]

    def get_tile_extents(self):

        def temp_total_bounds():
            bounds = self.gdf.geometry.bounds
            minx = bounds['minx'].min()
            miny = bounds['miny'].min()
            maxx = bounds['maxx'].max()
            maxy = bounds['maxy'].max()
            return minx, miny, maxx, maxy

        minx, miny, maxx, maxy = self.gdf.geometry.total_bounds
        minx, miny, maxx, maxy = temp_total_bounds()
        poly_coords = [(minx, miny), (minx, maxy), 
                       (maxx, maxy), (maxx, miny)]
        return Polygon(poly_coords)


def set_env_vars(env_name):
    user_dir = os.path.expanduser('~')
    path_parts = ('AppData', 'Local', 
                  'Continuum', 'anaconda3')
    path_parts = ('AppData', 'Local', 
                  'conda', 'conda')
    conda_dir = Path(user_dir).joinpath(*path_parts)
    env_dir = conda_dir / 'envs' / env_name
    share_dir = env_dir / 'Library' / 'share'
    script_path = conda_dir / 'Scripts'
    gdal_data_path = share_dir / 'gdal'
    proj_lib_path = share_dir

    if script_path.name not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + str(script_path)
    os.environ['GDAL_DATA'] = str(gdal_data_path)
    os.environ['PROJ_LIB'] = str(proj_lib_path)


if __name__ == '__main__':
    set_env_vars('shoreline')

    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)

    schema_path = Path(r'.\shoreline_schema.json')
    schema = Schema(schema_path)

    slt = ShorelineTile(arcpy.GetParameterInfo(), schema)
    shps = [Path(shp) for shp in slt.shp_paths.exportToString().split(';')]
    num_shps = len(shps)

    for i, shp in enumerate(shps, 1):

        arcpy.AddMessage('{} ({} of {})...'.format(shp.name, i, num_shps))
        slt.populate_gdf(shp)

        if not slt.gdf.empty:

            # determine overlapping state NOAA regions
            state_regions = slt.get_overlapping_state_regions()

            # apply state-region attributes
            slt.apply_state_region_attributes(state_regions)

            # apply tile-wide attributes
            slt.apply_tile_attributes()

            # output attributed gdf
            out_path = Path(slt.out_dir.value) / 'TEST_OUTPUT_FINAL.shp'
            slt.export(out_path)
