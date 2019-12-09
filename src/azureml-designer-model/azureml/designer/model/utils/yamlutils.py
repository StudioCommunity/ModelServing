from collections import defaultdict
import os

from ruamel.yaml import YAML, ruamel, RoundTripRepresenter


# ruamel supports dumping most of built-in python classes to yaml file,
# but with some exceptions. Add the missing ones here.
# NOTE: use add_multi_representer to add to super classes.
def register_yaml_representer(cls, representer):
    RoundTripRepresenter.add_representer(cls, representer)


# Set default supported classes which can dumped as yaml
register_yaml_representer(defaultdict, RoundTripRepresenter.represent_dict)

# A general entry for yaml operations.
# Put at top level so that we can share the instance across the project.
yaml = YAML()


def dump_to_yaml(obj, stream):
    ruamel.yaml.round_trip_dump(obj, stream)


def dump_to_yaml_file(obj, filename):
    with open(filename, 'w') as fout:
        dump_to_yaml(obj, fout)


def load_yaml(stream):
    return ruamel.yaml.safe_load(stream)


def load_yaml_file(filename):
    with open(filename, 'r') as fin:
        return load_yaml(fin)
