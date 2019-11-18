import os
import sys

class Config(object):
    """a class to storage the basic parameter"""

    def __init__(self):
        #work path
        self.workPath = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN'
        self.dataPath = self.workPath + '/data'