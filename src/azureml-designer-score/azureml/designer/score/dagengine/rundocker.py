import sys
import os
import subprocess


def main():
    abspath = os.getcwd()
    tag = abspath.rpartition('\\')[-1]
    base_dir = '/' + abspath.replace(':', '').replace('\\', '/')
    build(tag)
    run(tag, base_dir)


def build(tag):
    dockercommand = f'docker build --rm -t {tag} .'
    print(dockercommand)
    subprocess.call(dockercommand, shell=True)


def run(tag, base_dir):
    # base_dir = '/d/Repos/StudioCore/Product/Source/StudioCoreService/DeploymentService/PackageRoot/Data/Resources/amlstudio-ab59527bec2746aaad7d80'
    dockercommand = f'docker run --rm -it -v {base_dir}/input:/app/input -v {base_dir}/output:/app/output -v {base_dir}/data:/app/data -v {base_dir}/studiomodelpackage:/app/studiomodelpackage {tag} python main.py'
    print(dockercommand)
    subprocess.call(dockercommand, shell=True)


if __name__ == '__main__':
    main()
