import os
from os.path import dirname, abspath


def _substitute(root_path, top_module_name):
    dir_names = set()
    py_file_names = set()
    for name in os.listdir(root_path):
        entry_path = os.path.join(root_path, name)
        if os.path.isdir(entry_path):
            dir_names.add(name)
        elif entry_path.endswith(".py"):
            py_file_names.add(name)
    if "__init__.py" not in py_file_names:
        return

    for filename in py_file_names:
        file_path = os.path.join(root_path, filename)
        file_path = file_path.replace("\\", "/")
        rel_path_to_top = file_path[len(top_path) + 1:]
        module_hierarchy = [top_module_name] + rel_path_to_top.split('/')[:-1]
        print(f"file_path = {file_path}")
        print(f"module_hierarchy = {module_hierarchy}")
        converted_lines = []
        with open(file_path, 'r') as fp:
            for line in fp.readlines():
                if line.startswith("from ."):
                    parts = line.split(' ')
                    rel_module = parts[1]
                    leading_periods_cnt = 0
                    while leading_periods_cnt < len(rel_module) and rel_module[leading_periods_cnt] == '.':
                        leading_periods_cnt += 1
                    rel_module_suffix = rel_module[leading_periods_cnt:]
                    abs_module_prefix = '.'.join(module_hierarchy[: len(module_hierarchy) - leading_periods_cnt + 1])
                    if not rel_module_suffix:
                        abs_module = abs_module_prefix
                    else:
                        abs_module = abs_module_prefix + '.' + rel_module_suffix
                    parts[1] = abs_module
                    converted_line = ' '.join(parts)
                    print(f"{line.strip()} ---> {converted_line.strip()}")
                    converted_lines.append(converted_line)
                else:
                    converted_lines.append(line)

        with open(file_path, 'w') as fp:
            fp.writelines(converted_lines)

    for dir_name in dir_names:
        _substitute(os.path.join(root_path, dir_name), top_module_name)


def substitute(top_path):
    top_module_name = os.path.split(top_path)[-1]
    _substitute(top_path, top_module_name)


if __name__ == "__main__":
    PROJECT_ROOT_PATH = dirname(dirname(abspath(__file__)))
    top_path = os.path.join(PROJECT_ROOT_PATH, "src/azureml-designer-model/azureml")
    substitute(top_path)
