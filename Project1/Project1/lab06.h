#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>

void OddEvenSequential(std::vector<int>& arr)
{
	bool sorted = false;
	while (!sorted)
	{
		sorted = true;
		for (int i = 1; i < arr.size() - 1; i += 2)
		{
			if (arr[i] > arr[i + 1])
			{
				std::swap(arr[i], arr[i + 1]);
				sorted = false;
			}
		}

		for (int i = 0; i < arr.size() - 1; i += 2)
		{
			if (arr[i] > arr[i + 1])
			{
				std::swap(arr[i], arr[i + 1]);
				sorted = false;
			}
		}
	}
}

void MPI_OddEvenScatterGather(std::vector<int>& arr, int rank, int size)
{
	int n = arr.size();
	int local_n = n / size;
	std::vector<int> local_arr(local_n);
	MPI_Scatter(arr.data(), local_n, MPI_INT, local_arr.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

	OddEvenSequential(local_arr);

	for (int phase = 0; phase < size; phase++)
	{
		int partner = (phase % 2 == 0) ? rank ^ 1 : rank ^ 0;

		if (partner >= 0 && partner < size)
		{
			std::vector<int> recv_arr(local_arr);
			MPI_Sendrecv(local_arr.data(), local_n, MPI_INT, partner, 0, recv_arr.data(), local_n, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			std::vector<int> merged_arr(local_n * 2);
			std::merge(local_arr.begin(), local_arr.end(), recv_arr.begin(), recv_arr.end(), merged_arr.begin());

			if (rank < partner)
			{
				std::copy(merged_arr.begin(), merged_arr.begin() + local_n, local_arr.begin());
			}
			else
			{
				std::copy(merged_arr.begin() + local_n, merged_arr.end(), local_arr.begin());
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Gather(local_arr.data(), local_n, MPI_INT, arr.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
}

void MPI_OddEvenReplace(std::vector<int>& arr, int rank, int size)
{
	int n = arr.size();
	int local_n = n / size;
	std::vector<int> local_arr(local_n);
	MPI_Scatter(arr.data(), local_n, MPI_INT, local_arr.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

	OddEvenSequential(local_arr);

	for (int phase = 0; phase < size; phase++)
	{
		if ((rank + phase) % 2 == 0)
		{
			if (rank + 1 < size)
			{
				MPI_Sendrecv_replace(local_arr.data(), local_n, MPI_INT, rank + 1, 0, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		else
		{
			if (rank > 0)
			{
				MPI_Sendrecv_replace(local_arr.data(), local_n, MPI_INT, rank - 1, 0, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Gather(local_arr.data(), local_n, MPI_INT, arr.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
}

void compareAndExchange(int rank, int partner, std::vector<int>& local_data, int size, int sorted)
{
	std::vector<int> recv_data(local_data.size());

	MPI_Sendrecv(local_data.data(), local_data.size(), MPI_INT, partner, 0, recv_data.data(),
		local_data.size(), MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	if (recv_data[0] > local_data[local_data.size() - 1])
	{
		MPI_Sendrecv_replace(local_data.data(), local_data.size(), MPI_INT, partner, 0, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		sorted = 0;
	}
}

void MPI_OddEvenCompareExchange(std::vector<int>& arr, int rank, int size)
{
	int sorted = 0;
	while (!sorted)
	{
		sorted = 1;
		if (rank % 2 == 0 && rank + 1 < size)
		{
			compareAndExchange(rank, rank + 1, arr, size, sorted);
		}
		else if (rank > 0)
		{
			compareAndExchange(rank, rank - 1, arr, size, sorted);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		if (rank % 2 == 1 && rank + 1 < size)
		{
			compareAndExchange(rank, rank + 1, arr, size, sorted);
		}
		else if (rank > 0)
		{
			compareAndExchange(rank, rank - 1, arr, size, sorted);
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// reduce: unire / preluare a tuturor informatiiloe
		// all: syncs processes
		MPI_Allreduce(MPI_IN_PLACE, &sorted, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	}
}

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	const int N = 10'000;

	std::vector<int> data(N);
	if (rank == 0)
	{
		srand(time(NULL) * 0 + 1234);
		for (int i = 0; i < N; i++)
		{
			data[i] = rand();
		}
	}

	double time = MPI_Wtime();

	//if (rank == 0)
	//{
		//OddEvenSequential(data);
	//}

	//MPI_OddEvenScatterGather(data, rank, size);
	MPI_OddEvenReplace(data, rank, size);

	time = MPI_Wtime() - time;
	if (rank == 0)
	{
		std::cout << "sort took " << time << " seconds.\n";
	}


	MPI_Finalize();

	return 0;
}