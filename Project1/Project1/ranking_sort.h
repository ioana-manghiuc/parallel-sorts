#include <mpi.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdlib>

void MPI_RankingSort(std::vector<int>& data, int rank, int size)
{
    int n = data.size();
    int local_n = n / size;

    std::vector<int> full_data(n);
    std::vector<int> ranking(local_n);
    std::vector<int> overall_ranking(n);

    MPI_Bcast(data.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    full_data = data;

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

    MPI_Gather(ranking.data(), local_n, MPI_INT,
        overall_ranking.data(), local_n, MPI_INT,
        0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::vector<int> sorted(n);
        for (int i = 0; i < n; ++i) 
        {
            sorted[overall_ranking[i]] = full_data[i];
        }
        data = sorted;
    }
}