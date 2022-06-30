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
@click.argument("in_ndsexp", type=click.Path(exists=True))
@click.argument("out_jsonl")
@click.option("--min-support", default=2)
@click.option("--min-patt-size", default=1)
@click.option("--max-patt-size", default=int(10e99))
@click.option("--weighted", is_flag=True, default=False)
@click.option("--cache-dir", default=None)
def run_freqt(
    in_ndsexp,
    out_jsonl,
    min_support: int,
    min_patt_size: int,
    max_patt_size: int,
    weighted: bool,
    cache_dir: str | None,
):
    """Python implementation of Freqt algorithm (Asai, 2001) with similar CLI interface.

    Trees (represented as s-expressions) are streamed from STDIN. Mined patterns are
    written out to STDOUT.

    Differently than the original Freqt CLI, ours formats pattern data as JSON instead of XML.
    """

    max_trees = None  # 10_000

    def make_criterion_fn():

        from blind_timber.dlg import TreebankADLG

        metric = TreebankADLG()
        logger.info("Infering treebank statistics from input ndsexp")
        metric.fit_from_file(Path(in_ndsexp))

        def criterion_fn(patt_toks: str, support: int, n_nodes: int, out: dict):
            # If last node in pattern is one of the following, drop.
            if patt_toks[-1] in ("deprel:punct->W", "deprel:det->W"):
                return False

            if n_nodes > 5:
                adlg = metric([(" ".join(patt_toks), support)])[-1].item()
                if adlg < 0:
                    return False
                if n_nodes >= min_patt_size:
                    if adlg < 30:
                        return False
                out["adlg"] = round(adlg, 5)

            return True

        return criterion_fn

    freqt = FREQTOriginal(
        min_support=min_support,
        min_nodes=min_patt_size,
        max_nodes=max_patt_size,
        cache_dir=Path(cache_dir) if cache_dir else None,
        weighted=weighted,
        custom_criterion=make_criterion_fn(),
    )
    # original: 11_693_846
    logger.info("Indexing trees...")
    freqt.index_trees(it.islice(open(in_ndsexp), max_trees), total=max_trees)
    logger.info("Initial tree indexing complete.")
    patt: SubtreeDict
    with open(out_jsonl, mode="wt") as out_fp:
        for patt in freqt.iter_subtrees():
            out_fp.write(f"{ujson.dumps(patt)}\n")
            # out_fp.write(f"{patt['pattern']}\n")


if __name__ == "__main__":
    run_freqt()
