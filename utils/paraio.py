from .lazyio import LazyFileReader
from typing import List, Callable, Tuple

import multiprocessing as mp


# this function should be invoked under __main__ module
# func parameter requires the function only takes one FileReader type
# object as parameter

# use ordered pool instead, change it later
def job_distribute(
    file_chunks: List[LazyFileReader],
    func: Callable
):
    
    workers = []
    res_collector = mp.Manager().list()
    for i, chunk in enumerate(file_chunks):
        worker = mp.Process(target=func, args=(i, chunk, res_collector))
        worker.start()
        workers.append(worker)
    
    # wait for jobs to complete
    for worker in workers:
        worker.join()
    return res_collector
