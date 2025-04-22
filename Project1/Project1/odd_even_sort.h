#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>

void compareAndExchange(int rank, int partner, std::vector<int>& local_data, int& sorted_flag) {
	int local_size = local_data.size();
	std::vector<int> recv_data(local_size);

	MPI_Sendrecv(local_data.data(), local_size, MPI_INT, partner, 0,
		recv_data.data(), local_size, MPI_INT, partner, 0,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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
}

void MPI_OddEven(std::vector<int>& arr, int rank, int size)
{
	int n = arr.size();
	int local_n = n / size;

	std::vector<int> local_arr(local_n);
	MPI_Scatter(arr.data(), local_n, MPI_INT, local_arr.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

	std::sort(local_arr.begin(), local_arr.end());

	int sorted = 0;
	while (!sorted)
	{
		sorted = 1;

		if (rank % 2 == 0 && rank + 1 < size)
		{
			compareAndExchange(rank, rank + 1, local_arr, sorted);
		}
		else if (rank % 2 == 1 && rank - 1 >= 0)
		{
			compareAndExchange(rank, rank - 1, local_arr, sorted);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		if (rank % 2 == 1 && rank + 1 < size)
		{
			compareAndExchange(rank, rank + 1, local_arr, sorted);
		}
		else if (rank % 2 == 0 && rank - 1 >= 0)
		{
			compareAndExchange(rank, rank - 1, local_arr, sorted);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &sorted, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	}

	MPI_Gather(local_arr.data(), local_n, MPI_INT, arr.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
}