import time

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"Dummy module sleeping")
logger = logging.getLogger(__name__)


# ignore extra arguments
@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--sleep_time_in_seconds', default=5)
def run_pipeline(sleep_time):
    print(f'Sleep time in seconds: {sleep_time}')
    print('Slept at: %s' % time.ctime())
    time.sleep(sleep_time)
    print('Woke up at: %s' % time.ctime())

# python -m azureml.studio.score.dummy_module --sleep_time_in_seconds 5
if __name__ == '__main__':
    run_pipeline()