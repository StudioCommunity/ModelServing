import os
import os.path
import sys
from git import Repo 
import shutil
import tempfile

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

from ..logger import get_logger
logger = get_logger(__name__)
logger.info(f"Import Folder from git")
        

@click.command()
@click.option('--git_url')
@click.option('--out_path', default='out')
def run_pipeline(git_url, out_path):
    temp_folder = tempfile.TemporaryDirectory()
    temp_path = temp_folder.name
    #temp_path = 'temp'
    print(f'Cloning {git_url} to {temp_path}')
    repo = Repo.clone_from(git_url, temp_path)
    root = repo.working_dir
    print(f'Cloned to {root}')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for entry in os.scandir(root):
        if entry.name == '.git':
            continue
        src = os.path.join(root, entry.name)
        print(f'Copying {src} to {out_path}')
        shutil.copy2(src, out_path)
    #temp_folder.cleanup()
    print(f"OUTPUT({out_path}): {os.listdir(out_path)}")
    
# python -m dstest.importer.git2folder --git_url https://github.com/RichardZhaoW/Models.git --out_path out
if __name__ == '__main__':
    run_pipeline()