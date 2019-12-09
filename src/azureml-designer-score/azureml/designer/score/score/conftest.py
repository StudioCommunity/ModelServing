import os


def pytest_ignore_collect(path, config):
    """ return True to prevent considering this path for collection.
    This hook is consulted for all files and directories prior to calling
    more specific hooks.
    """
    if os.path.split(path)[-1] == "InputPort1":
        return True
