import sys
from dragon.workflows.batch import Batch
from dragon.native.process import ProcessTemplate
from dragon.infrastructure.facts import PMIBackend
from dragon.data import DDict

def queue_job(batch, nranks, cfl=0.8, dser=""):

    rank_tmpls = [(nranks, ProcessTemplate(target=sys.executable,
                                           args=("../../course2/orchestrating_MPI/mpi4py_example.py","--cfl",cfl,"--dser",dser)))]
    job = batch.options(pmi=PMIBackend.PMIX).job(process_templates=rank_tmpls)
    print(f"Got a job {job}", flush=True)
    return job.uid, job


def main():

    ddict = DDict(total_mem=(16 * 1024**2))
    batch = Batch()

    mpijobs = []
    for _ in range(1):
        uid, job = queue_job(batch, 2, 0.8, ddict.serialize())
        mpijobs.append(job)

    for job in mpijobs:
        try:
            ecodes = job.get()
            print(f"Got ecodes: {ecodes}", flush=True)
        except Exception as e:
            print(f"Got exception from MPI job: {e}", flush=True)

    keys = [k for k in ddict.keys()]
    print(f"KEYS = {keys}", flush=True)

    batch.join()
    batch.destroy()
    ddict.destroy()

if __name__ == '__main__':
    main()