from ..logger import get_logger

logger = get_logger(__name__)

class MergeError(Exception):
    pass


def merge_envs(envs):
    if not envs:
        return None

    names = [env["name"] for env in envs if env and "name" in env]
    channels_list = [env["channels"] for env in envs if env and "channels" in env]
    dependencies_list = [env["dependencies"] for env in envs if env and "dependencies" in env]

    merged_name = merge_names(names)
    merged_channels = merge_channels_lists(channels_list)
    merged_dependencies = merge_dependencies_lists(dependencies_list)

    merged_env = {}
    if merged_name:
        merged_env["name"] = merged_name
    if merged_channels:
        merged_env["channels"] = merged_channels
    if merged_dependencies:
        merged_env["dependencies"] = merged_dependencies

    return merged_env


def merge_names(names: list):
    for name in names:
        if name:
            return name


def merge_channels_lists(channels_list: list):
    if not channels_list:
        return None

    merged_channels = channels_list[0]
    merged_channels_set = set(channels_list[0])
    for channels in channels_list[1:]:
        for channel in channels:
            if channel not in merged_channels_set:
                merged_channels.append(channel)
                merged_channels_set.add(channel)

    return merged_channels


def merge_dependencies_lists(dependencies_list):
    """
    Merge all dependencies to one list and return it.
    Two overlapping dependencies (e.g. package-a and package-a=1.0.0) are not
    unified, and both are left in the list (except cases of exactly the same
    dependency). Conda itself handles that very well so no need to do this ourselves,
    unless you want to prettify the output by hand.
    :param dependencies_list: list of lists dependencies
    :return: merged dependencies list
    """
    pip_dependencies = []
    non_pip_dependencies_set = set()
    for dependencies in dependencies_list:
        for dependency in dependencies:
            if isinstance(dependency, dict) and "pip" in dependency:
                pip_dependencies.append(dependency)
            elif dependency not in non_pip_dependencies_set:
                non_pip_dependencies_set.add(dependency)
    merged_pip = merge_pips(pip_dependencies)
    return sorted(list(non_pip_dependencies_set)) + [merged_pip]


def merge_pips(pip_list: list):
    """
    Simply concat all pip lists
    :param pip_list:
    :return:
    """
    return {"pip": sorted(sum([pip_dep["pip"] for pip_dep in pip_list], []))}

