import json
import geopandas as gpd
from pathlib import Path


class SchemaAttribute:

    def __init__(self, label, **kwargs):
        self.attribute_label = label
        self.__dict__.update(kwargs)

    def __str__(self):
        return json.dumps(self.__dict__, indent=1)


class Shoreline:

    def __init__(self):
        pass


def create_schema():
    schema_path = Path(r'Z:\ShoreGen\ShoreGen_noaa-rsd\shoreline_schema.json')
    with open(schema_path, 'r') as j:
        schema_definition = json.load(j)

    schema = {}
    for k, v in schema_definition.items():
        schema[k] = SchemaAttribute(k, **v)

    return schema


def main():
    schema = create_schema()
    print(schema)

    proj_dir = Path(r'\\ngs-s-rsd\Lidar_Contract00\TX1803\Imagery\ortho\shp')
    shp_dir = proj_dir / 'tiles'

    shps = list(shp_dir.glob('*.shp'))
    num_shps = len(shps)
    for i, shp in enumerate(shps, 1):
        print('processing {} ({} of {})...'.format(shp.name, i, num_shps))
        gdf = gpd.read_file(str(shp))

        if not gdf.empty:
            gpkg_path =  proj_dir / 'TX1803.gpkg'
            layer = 'shp_{}'.format(shp.stem)
            gdf.to_file(gpkg_path, layer=layer, driver='GPKG')


if __name__ == '__main__':
    main()
