"""CLI for RF DETR and RF Segment models - training, evaluation, and export."""
import click


@click.group()
def rf():
    """CLI for RF DETR and RF Segment models - training, evaluation, and export."""
    pass


# Import subgroups from separate modules
from cvsdk.rf.det import det as det_cli
from cvsdk.rf.seg import seg as seg_cli


# Add subgroups to the main rf group
rf.add_command(det_cli)
rf.add_command(seg_cli)
