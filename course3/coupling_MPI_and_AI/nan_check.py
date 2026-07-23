"""Minimal NaN check: run the CFD experiments once, then scan for NaNs.

This is a stripped-down version of ``agent_workflow.py`` with NO LLM agents
and NO orchestrator.  It reuses the deterministic building blocks defined
there:

* ``run_experiments_node`` — runs one (fake) CFD simulation per CFL value
  across ``num_ranks`` MPI ranks and writes each rank's result array into a
  shared Dragon Distributed Dictionary (DDict).
* ``make_scan_all_ranks`` — builds the ``scan_all_ranks`` tool that reads
  those arrays back and reports which CFLs were stable vs. produced NaNs.

Usage::

    dragon nan_check.py
"""

import dragon
import multiprocessing as mp

from dragon.data.ddict import DDict
from dragon.workflows.batch import Batch

from agent_workflow import (
    NUM_RANKS,
    make_scan_all_ranks,
    run_experiments_node,
)


def main(init_cfls, num_ranks):
    # Shared data store carrying CFL values + per-rank result arrays.
    data_store = DDict(1, 1, 2 * 1024 * 1024)
    batch = Batch(results_ddict_mem=int(10 * 1024 * 1024))

    try:
        # Seed the CFLs and run one experiment per rank.  Passing None as the
        # upstream marks this as the initial (user-seeded) run, so the node
        # reads cfl_values from the DDict instead of a generator's output.
        data_store["cfl_values"] = init_cfls
        run_experiments_node(batch, data_store.serialize(), num_ranks, None)

        # Scan every rank array for NaNs and print the report.  Called directly
        # in-process here (no agent/tool loop); the live DDict handle is used
        # as-is.
        scan_all_ranks = make_scan_all_ranks(data_store, num_ranks)
        result = scan_all_ranks()

        print("\n--- NaN scan report ---", flush=True)
        print(result["report"], flush=True)
    finally:
        batch.join()
        try:
            data_store.destroy()
        except Exception:
            pass
        print("[teardown] Data store (DDict) destroyed.", flush=True)


if __name__ == "__main__":
    mp.set_start_method("dragon")
    init_cfls = [0.3, 2.4, 9.9, 0.8]
    main(init_cfls=init_cfls, num_ranks=NUM_RANKS)
