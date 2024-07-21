import click

from bluetooth_demod.__about__ import __version__


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="Proton Payload")
def bluetooth_demod():
    click.echo("Hello world!")
