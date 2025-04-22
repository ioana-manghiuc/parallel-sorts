#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>

void compareAndMerge(int rank, int partner, std::vector<int>& local_data, int local_n) {
    std::vector<int> recv_data(local_n);

    MPI_Sendrecv(local_data.data(), local_n, MPI_INT, partner, 0,
        recv_data.data(), local_n, MPI_INT, partner, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<int> merged(local_n * 2);
    std::merge(local_data.begin(), local_data.end(), recv_data.begin(), recv_data.end(), merged.begin());

    if (rank < partner) {
        std::copy(merged.begin(), merged.begin() + local_n, local_data.begin());
    }
    else {
        std::copy(merged.end() - local_n, merged.end(), local_data.begin());
    }
}

void MPI_ShellSort(std::vector<int>& data, int rank, int size) {
    int n = data.size();
    int local_n = n / size;

    std::vector<int> local_data(local_n);
    MPI_Scatter(data.data(), local_n, MPI_INT, local_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    std::sort(local_data.begin(), local_data.end());

    int max_log = static_cast<int>(std::log2(size));

    // --- Stage 1: Divide into shells and exchange extremes
    for (int l = 0; l <= max_log; ++l) {
        int stride = 1 << (max_log - l);
        int group_id = rank / stride;

        int partner = (group_id % 2 == 0) ? rank + stride : rank - stride;

        if (partner >= 0 && partner < size) {
            compareAndMerge(rank, partner, local_data, local_n);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- Stage 2: Odd-Even Refinement Phase
    int global_sorted = 0;
    while (!global_sorted) {
        int local_sorted = 1;

        if ((rank % 2 == 0) && (rank + 1 < size)) {
            compareAndMerge(rank, rank + 1, local_data, local_n);
        }
        else if ((rank % 2 == 1) && (rank - 1 >= 0)) {
            compareAndMerge(rank, rank - 1, local_data, local_n);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if ((rank % 2 == 1) && (rank + 1 < size)) {
            compareAndMerge(rank, rank + 1, local_data, local_n);
        }
        else if ((rank % 2 == 0) && (rank - 1 >= 0)) {
            compareAndMerge(rank, rank - 1, local_data, local_n);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // All processes agree if sorted
        MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    MPI_Gather(local_data.data(), local_n, MPI_INT, data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
}