#include<iostream>

#include<fstream>
#include<vector>

#include "direct_sort.h"
#include "bucket_sort.h"

int main(int argc, char** argv)
{
	int n = 1'000;
	std::ifstream small_data_file("small_data.txt", std::ifstream::in);

	std::vector<int>smallData(n);
	for (size_t i = 0; i < n; i++)
	{
		small_data_file >> smallData[i];
	}

	small_data_file.close();

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double time = MPI_Wtime();

	MPI_BucketSort(smallData, rank, size);

	time = MPI_Wtime() - time;
	if (rank == 0)
	{
		std::cout << "sort took " << time << " seconds.\n";
		std::cout << std::is_sorted(smallData.begin(), smallData.end());
	}


	MPI_Finalize();

	return 0;
}