## Minimal 3-node example of PyPSA power flow using MPI for parallel execution

import pypsa
import numpy as np

import dill

from mpi4py import MPI
MPI.pickle.__init__(dill.dumps, dill.loads)

comm = MPI.COMM_WORLD
numprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
procname = MPI.Get_processor_name()

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

print('Rank:', rank, 'line flows:', network.lines_t.p0[:5], '\n')

# writing output in parallel
#network.export_to_netcdf("network.nc")

endtime = MPI.Wtime()

print('Total time: {0:.2f} sec'.format(endtime-starttime, ))


