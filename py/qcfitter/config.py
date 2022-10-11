from configparser import ConfigParser
from pkg_resources import resource_filename

PKG_DEFAULT_INI = resource_filename('qcfitter', 'default-config.ini')

class Config(object):
    """docstring for Config"""
    def __init__(self, filename):
        self.config = ConfigParser()
        self.config.read(PKG_DEFAULT_INI)
        self.config.read(filename)
        # self.filename = filename
        