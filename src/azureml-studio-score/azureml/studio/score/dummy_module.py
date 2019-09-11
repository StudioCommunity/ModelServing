from datetime import datetime
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
@click.option('--sleep_time_milli_seconds', default=1.0)
def run_pipeline(sleep_time_milli_seconds):
    print(f'Sleep time in seconds: {sleep_time_milli_seconds}')
    current_milli_time = lambda: int(round(time.time() * 1000))
    print('Slept at: %s' % current_milli_time())
    time.sleep(sleep_time_milli_seconds)
    print('Woke up at: %s' % current_milli_time())

# python -m azureml.studio.score.dummy_module --sleep_time_milli_seconds 0.1
if __name__ == '__main__':
    run_pipeline()
