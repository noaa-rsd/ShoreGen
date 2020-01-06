import json
import os
from pathlib import Path


class Schema:

    def __init__(self, path):
        self.path = path
        self.set_attributes()
        self.atypes = {
            'tile': [
                'SOURCE_ID',
                'SRC_DATE',
                'HOR_ACC',
                'INFORM',
                'VER_DATE',
                'SRC_RESOLU',
                'DATA_SOURC',
                'EXT_METH',
                'DAT_SET_CR'
                ],
            'line': [
                'ATTRIBUTE',
                'FIPS_ALPHA',
                'NOAA_Regio'
                ]
            }


    def __str__(self):
        return json.dumps(self.__dict__, indent=1)

    def set_attributes(self):
        with open(self.path, 'r') as j:
            self.__dict__ = json.load(j)


if __name__ == '__main__':
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)

    schema_path = Path(r'.\shoreline_schema.json')

    schema = Schema(schema_path)
    print(schema.atypes)
