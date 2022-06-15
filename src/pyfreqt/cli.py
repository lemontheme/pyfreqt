import sys
import logging
import itertools as it
from pathlib import Path

import click
import ujson

from .freqt import FREQTOriginal, SubtreeDict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument("in_ndsexp", type=click.File("rt"))
@click.argument("out_jsonl", type=click.File("wt"))
@click.option("--min-support", default=2)
@click.option("--min-patt-size", default=1)
@click.option("--max-patt-size", default=int(10e99))
@click.option("--cache-dir", default=None)
def run_freqt(
    in_ndsexp,
    out_jsonl,
    min_support: int,
    min_patt_size: int,
    max_patt_size: int,
    cache_dir: str | None,
):
    """Python implementation of Freqt algorithm (Asai, 2001) with similar CLI interface.

    Trees (represented as s-expressions) are streamed from STDIN. Mined patterns are
    written out to STDOUT.

    Differently than the original Freqt CLI, ours formats pattern data as JSON instead of XML.
    """
    freqt = FREQTOriginal(
        min_support=min_support,
        min_nodes=min_patt_size,
        max_nodes=max_patt_size,
        # cache_dir=Path(cache_dir) if cache_dir else None,
        weighted=True
    )
    # original: 11_693_846
    max_trees = 1_000_000
    freqt.index_trees(it.islice(in_ndsexp, max_trees), total=max_trees)
    logger.info("Initial tree indexing complete.")
    patt: SubtreeDict
    for patt in freqt.iter_subtrees():
        # sys.stdout.write(f"{ujson.dumps(patt)}\n")
        out_jsonl.write(f"{patt['pattern']}\n")


if __name__ == "__main__":
    run_freqt()