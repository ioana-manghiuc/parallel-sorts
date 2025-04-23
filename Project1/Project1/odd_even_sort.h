#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>

void compareAndExchange(int rank, int partner, std::vector<int>& local_data, int& sorted_flag,
						double& computation_time, double& communication_time) 
{
	double start_time = 0.0;
	int local_size = local_data.size();
	std::vector<int> recv_data(local_size);

	start_time = MPI_Wtime();
	MPI_Sendrecv(local_data.data(), local_size, MPI_INT, partner, 0,
		recv_data.data(), local_size, MPI_INT, partner, 0,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	communication_time += MPI_Wtime() - start_time;

	start_time = MPI_Wtime();
	std::vector<int> merged(local_size * 2);
	std::merge(local_data.begin(), local_data.end(), recv_data.begin(), recv_data.end(), merged.begin());

	std::vector<int> original = local_data;

	if (rank < partner) {
		std::copy(merged.begin(), merged.begin() + local_size, local_data.begin());
	}
	else {
		std::copy(merged.end() - local_size, merged.end(), local_data.begin());
	}

	if (local_data != original) {
		sorted_flag = 0;
	}

	computation_time += MPI_Wtime() - start_time;
}

void MPI_OddEvenSort(std::vector<int>& arr, int rank, int size, double& computation_time, double& communication_time)
{
	double start_time = 0.0;
	int n = arr.size();
	int local_n = n / size;
	std::vector<int> local_arr(local_n);

	start_time = MPI_Wtime();
	MPI_Scatter(arr.data(), local_n, MPI_INT, local_arr.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
	communication_time += MPI_Wtime() - start_time;

	start_time = MPI_Wtime();
	std::sort(local_arr.begin(), local_arr.end());
	computation_time += MPI_Wtime() - start_time;

	int is_sorted = 0;
	while (!is_sorted)
	{
		is_sorted = 1;

		if (rank % 2 == 0 && rank + 1 < size)
		{
			compareAndExchange(rank, rank + 1, local_arr, is_sorted, communication_time, computation_time);
		}
		else if (rank % 2 == 1 && rank - 1 >= 0)
		{
			compareAndExchange(rank, rank - 1, local_arr, is_sorted, communication_time, computation_time);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		if (rank % 2 == 1 && rank + 1 < size)
		{
			compareAndExchange(rank, rank + 1, local_arr, is_sorted, communication_time, computation_time);
		}
		else if (rank % 2 == 0 && rank - 1 >= 0)
		{
			compareAndExchange(rank, rank - 1, local_arr, is_sorted, communication_time, computation_time);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		start_time = MPI_Wtime();
		MPI_Allreduce(MPI_IN_PLACE, &is_sorted, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		communication_time += MPI_Wtime() - start_time;
	}
	start_time = MPI_Wtime();
	MPI_Gather(local_arr.data(), local_n, MPI_INT, arr.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
	communication_time += MPI_Wtime() - start_time;
}