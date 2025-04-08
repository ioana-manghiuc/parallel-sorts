#include <iostream>
#include "lab05_sorting_methods.h"

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// vector with N elements and N dim
	const int N = 1'000'000;
	double time;
	std::vector<int> data(N);

	// process with rank 0 + initialize vector

	if (rank == 0)
	{
		for (int i = 0; i < data.size(); i++)
		{
			data[i] = rand();
		}
	}

	// time before sorting
	time = MPI_Wtime();

	// sort vector
	//MPI_Sort(data, rank, size, &merge_sort);
	//MPI_Sort(data, rank, size, &bubble_sort);
	//MPI_Sort(data, rank, size, &selection_sort);
	//MPI_Sort(data, rank, size, &insertion_sort);
	//MPI_Sort(data, rank, size, &quick_sort);
	MPI_Sort(data, rank, size, &heap_sort);

	// time after sorting
	time = MPI_Wtime() - time;

	if (rank == 0)
	{
		std::cout << "sorting took " << time << " seconds.\n";
	}

	MPI_Finalize();

	return 0;
}