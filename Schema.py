import json


class Schema:

    def __init__(self, path):
        self.path = path
        self.set_attributes()

    def __str__(self):
        return json.dumps(self.__dict__, indent=1)

    def set_attributes(self):
        with open(self.path, 'r') as j:
            self.__dict__ = json.load(j)


if __name__ == '__main__':
    pass
