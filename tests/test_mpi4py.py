from mpi4py import MPI
COMM = MPI.COMM_WORLD
import numpy as np
import pytest

from simemc import mpi_utils
print = mpi_utils.printRf


@pytest.mark.mpi(min_size=2)
def test_send_recv_pattern():
    """
    checks the mpi send/recv of numpy arrays, same shape as Pilatus detector
    this form of communication occurs in each emc iteration
    when computing te tomograms from images that exist on separate MPI ranks
    """
    assert COMM.size > 1
    np.random.seed(COMM.rank)
    nsend = 25
    img_sh = 1,2527, 2463
    data = np.random.random(( nsend,) + img_sh)
    possible_dests = [i for i in range(COMM.size) if i != COMM.rank]
    dest_rank = np.random.choice(possible_dests, replace=True, size=nsend)
    source_dest = list(zip([COMM.rank]*nsend, dest_rank))


    # check all data sums that will be sent to each rank
    data_to_each_rank= []
    for rank in range(COMM.size):
        data_to_rank = np.zeros(img_sh)
        if rank in list(dest_rank):
            data_to_rank = data[dest_rank==rank].sum(axis=0)
        data_to_rank = COMM.bcast(COMM.reduce(data_to_rank))

        data_to_each_rank.append(data_to_rank)
            

    all_source_dest = COMM.reduce(source_dest)
    send_to = None
    receive_from = None
    if COMM.rank==0:
        receive_from = {}
        send_to = {}
        for i, (s,d) in enumerate(all_source_dest):
            tag = i
            if d not in receive_from:
                receive_from[d] = []
            receive_from[d].append((s, tag))
            if s not in send_to:
                send_to[s] = []
            send_to[s].append((d, tag))

    send_to = COMM.bcast(send_to)
    receive_from = COMM.bcast(receive_from)
    COMM.barrier()

    sent_req = []
    if COMM.rank in send_to:
        for i, (dest,tag) in enumerate(send_to[COMM.rank]):
            print("sent packet rank %d -> rank %d !" % (COMM.rank, dest), flush=True)
            req =COMM.isend(data[i], dest=dest, tag=tag)
            sent_req.append(req)
            
    packet_data = np.zeros(img_sh)
    if COMM.rank in receive_from:
        for i, (source, tag) in enumerate(receive_from[COMM.rank]):
            packet = COMM.recv(source=source, tag=tag)
            packet_data += packet
    for req in sent_req:
        req.wait()

    COMM.barrier()

    for rank in range(COMM.size):
        if COMM.rank==rank:
            assert np.allclose(packet_data, data_to_each_rank[rank])
            print("OK")

    COMM.barrier()
    mpi_utils.print0f("All ranks ok!")


if __name__=="__main__":
    test_send_recv_pattern()

