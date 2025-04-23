#include<iostream>

#include<fstream>
#include<vector>

#include "direct_sort.h"
#include "bucket_sort.h"
#include "odd_even_sort.h"
#include "ranking_sort.h"
#include "shell_sort.h"


int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n = 10'000'000;
	std::ifstream data_file("data_10mil.txt", std::ifstream::in);

	std::vector<int>data(n);
	if (rank == 0)
	{
		for (size_t i = 0; i < n; i++)
		{
			data_file >> data[i];
		}

		std::cout << "Read " << data.size() << " elements from file.\n";
		data_file.close();
	}

	double computation_time = 0.0, communication_time = 0.0;
	double execution_time = MPI_Wtime();

	MPI_ShellSort(data, rank, size, computation_time, communication_time);

	execution_time = MPI_Wtime() - execution_time;
	if (rank == 0)
	{
		std::cout << "EXECUTION TIME: " << execution_time << "\n";
		std::cout << "COMPUTATION TIME: " << computation_time << "\n";
		std::cout << "COMMUNICATION TIME: " << communication_time << "\n";
		std::cout << std::is_sorted(data.begin(), data.end());
	}

	MPI_Finalize();

	return 0;
}