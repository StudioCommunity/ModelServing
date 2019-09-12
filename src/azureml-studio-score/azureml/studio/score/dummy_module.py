import time
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
@click.option('--sleep_time_milli_seconds', default=1000)
def run_pipeline(sleep_time_milli_seconds):
    print(f'Sleep time in milliseconds: {sleep_time_milli_seconds}')
    current_milli_time = lambda: int(round(time.time() * 1000))
    start_time = current_milli_time()
    print('Slept at: %s' % start_time)
    time.sleep(sleep_time_milli_seconds / 1000)
    end_time = current_milli_time()
    print('Woke up at: %s' % end_time)
    print(f'Sleep duration: {end_time - start_time}')


# python -m azureml.studio.score.dummy_module --sleep_time_milli_seconds 1000
if __name__ == '__main__':
    run_pipeline()
