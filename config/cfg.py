import os
import json
import torch
import argparse


class Config(object):
    def __init__(self, logger, args):
        self.logger = logger
        self.config = vars(args)
        
    def save_config(self, path):
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        self.logger.debug("Config saved to file {}".format(path))

    def load_config(self, path, verbose=True):
        with open(path) as f:
            self.config = json.load(f)

        self.logger.debug("Config loaded from file {}".format(path))

    def print_config(self):
        debug = "Running with the following configs:\n"
        for k,v in self.config.items():
            debug += "\t{} : {}\n".format(k, str(v))
        self.logger.debug("\n" + debug + "\n")

