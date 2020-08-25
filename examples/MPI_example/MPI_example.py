## Minimal 3-node example of PyPSA power flow using MPI for parallel execution

# Additional packages mpi4py, dill
# e.g.
# conda install dill
# conda install -c intel mpi4py

# Execute via mpiexec, mpirun, srun etc.
# e.g.
# mpiexec -n 4 python MPI_example.py


import pypsa
import numpy as np

import logging
logging.basicConfig(level=logging.ERROR)

import dill
from mpi4py import MPI
MPI.pickle.__init__(dill.dumps, dill.loads)

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

starttime = MPI.Wtime()

if rank == 0:
    network = pypsa.Network("network")
    network.set_snapshots(network.snapshots[:1000])

    # create rule to distribute workload
    def split_work():
        return np.array_split(network.snapshots, numprocs)

    # distribute tasks
    chunks = split_work()
    for pid, snapshots in enumerate(chunks[1:]):
        comm.send(network.copy(snapshots=snapshots), dest=pid+1, tag=11)
    network.set_snapshots(chunks[0])

else:
    network = comm.recv(source=0, tag=11)

# Perform Newton-Raphson power flow in parallel
network.pf()

## optionally gather partial solutions and merge them to initial network
#partial_networks = comm.gather(network, root=0)

print('Rank:', rank, '\nLine flows:', network.lines_t.p0[:3], '\n')

# writing output in parallel
#network.export_to_hdf5("network.h5")

endtime = MPI.Wtime()

print('Total time: {0:.2f} sec\n\n'.format(endtime-starttime, ))


