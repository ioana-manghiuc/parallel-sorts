#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>

bool compareAndMerge(int rank, int partner, std::vector<int>& local_data, int local_n, double& computation_time, double& communication_time) 
{
    double start_time = 0.0;
    std::vector<int> recv_data(local_n);

    start_time = MPI_Wtime();
    MPI_Sendrecv(local_data.data(), local_n, MPI_INT, partner, 0,
        recv_data.data(), local_n, MPI_INT, partner, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    communication_time += MPI_Wtime() - start_time;

    start_time = MPI_Wtime();
    std::vector<int> merged(local_n * 2);
    std::merge(local_data.begin(), local_data.end(), recv_data.begin(), recv_data.end(), merged.begin());
    
    std::vector<int> original = local_data;

    if (rank < partner) 
    {
        std::copy(merged.begin(), merged.begin() + local_n, local_data.begin());
    }
    else 
    {
        std::copy(merged.end() - local_n, merged.end(), local_data.begin());
    }
    computation_time += MPI_Wtime() - start_time;

    return local_data != original;
}

void MPI_ShellSort(std::vector<int>& data, int rank, int size, double& computation_time, double& communication_time) 
{
    double start_time = 0.0;
    int n = data.size();
    int local_n = n / size;

    std::vector<int> local_data(local_n);
    start_time = MPI_Wtime();
    MPI_Scatter(data.data(), local_n, MPI_INT, local_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - start_time;

    start_time = MPI_Wtime();
    std::sort(local_data.begin(), local_data.end());
    computation_time += MPI_Wtime() - start_time;

    int max_log = static_cast<int>(std::log2(size));

    for (int l = 0; l <= max_log; ++l) {
        int stride = 1 << (max_log - l);
        int group_id = rank / stride;

        int partner = (group_id % 2 == 0) ? rank + stride : rank - stride;

        if (partner >= 0 && partner < size) 
        {
            compareAndMerge(rank, partner, local_data, local_n, computation_time, communication_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    int global_sorted = 0;
    while (!global_sorted)
    {
        bool changed = false;

        if ((rank % 2 == 0) && (rank + 1 < size)) 
        {
            changed = changed || compareAndMerge(rank, rank + 1, local_data, local_n, computation_time, communication_time);
        }
        else if ((rank % 2 == 1) && (rank - 1 >= 0)) 
        {
            changed = changed || compareAndMerge(rank, rank - 1, local_data, local_n, computation_time, communication_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if ((rank % 2 == 1) && (rank + 1 < size)) 
        {
            changed = changed || compareAndMerge(rank, rank + 1, local_data, local_n, computation_time, communication_time);
        }
        else if ((rank % 2 == 0) && (rank - 1 >= 0)) 
        {
            changed = changed || compareAndMerge(rank, rank - 1, local_data, local_n, computation_time, communication_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        int local_sorted = changed ? 0 : 1;
        start_time = MPI_Wtime();
        MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        communication_time += MPI_Wtime() - start_time;
    }

    start_time = MPI_Wtime();
    MPI_Gather(local_data.data(), local_n, MPI_INT, data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - start_time;
}