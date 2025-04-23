#include <mpi.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdlib>

void MPI_RankingSort(std::vector<int>& data, int rank, int size, double& computation_time, double& communication_time)
{
    double start_time = 0.0;
    int n = data.size();
    int local_n = n / size;

    std::vector<int> full_data(n);
    std::vector<int> ranking(local_n);
    std::vector<int> overall_ranking(n);

    start_time = MPI_Wtime();
    MPI_Bcast(data.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - start_time;

    full_data = data;

    start_time = MPI_Wtime();
    for (int i = 0; i < local_n; ++i) 
    {
        int global_i = i + rank * local_n;
        ranking[i] = 0;
        for (int j = 0; j < n; ++j) 
        {
            if (full_data[j] < full_data[global_i]) 
            {
                ranking[i]++;
            }
        }
    }
    computation_time += MPI_Wtime() - start_time;

    start_time = MPI_Wtime();
    MPI_Gather(ranking.data(), local_n, MPI_INT, overall_ranking.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - start_time;

    if (rank == 0)
    {
        start_time = MPI_Wtime();
        std::vector<int> sorted(n);
        for (int i = 0; i < n; ++i) 
        {
            sorted[overall_ranking[i]] = full_data[i];
        }
        data = sorted;
        computation_time += MPI_Wtime() - start_time;
    }
}