import click
from esme.download import download_model


@click.command()
@click.option(
    '--model_name', '-m',
    required=True,
    type=str,
    help='The model name. Must be one of: esm1b, esm1v, esm1v_1, esm1v_2,'
    ' esm1v_3, esm1v_4, esm1v_5, esm2, esm2_8m, esm2_35m, esm2_150m,'
    ' esm2_650m, esm2_3b, esm2_15b, esmc, esmc_300m, esmc_600m'
)
@click.option(
    '--local_dir', '--dir', '-d',
    default=None,
    type=str,
    help='The directory to save the model to.'
)
def cli_download(model_name, local_dir=None):
    download_model(model_name, local_dir)
