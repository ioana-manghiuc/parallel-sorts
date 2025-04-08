#include<iostream>
#include<mpi.h>
#include <ctime>
#include<cstdlib>
#include<vector>

/*

ALGORITM TIP STRING MATCHING

Pentru a demonstra eficiența programarii paralele pe vectori am pregătit două dataseturi:
- Un fisier de 100 milioane caractere (95.3 MB) - pentru procesoarele mai lente
- Un fisier de 1 miliard caractere (953 MB) - pentru procesoarele mai rapide

*/

int main(int argc, char** argv)
{
	int rank, size;

	// parallel execution initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::vector<int> input_array;
	int n = 1'000'000;

	if (rank == 0)
	{
		for (int i = 0; i < n; i++)
		{
			input_array.push_back(i); // citire date, random, whatevs
		}
	}

	std::vector<int> scattered_array(n / size);

	MPI_Scatter(input_array.data(), n / size, MPI_INT, scattered_array.data(), n / size, MPI_INT, 0, MPI_COMM_WORLD);

	// SCENARIU 1 : SUMA ELEMENTELOR DIN VECTOR
	long long sum = 0, final_sum = 0;
	int min = INT_MAX, final_min = INT_MAX;
	int max = INT_MIN, final_max = INT_MIN;
	double time = MPI_Wtime();

	for (int i = 0; i < n / size; i++)
	{
		sum += scattered_array[i];
		min = std::min(min, scattered_array[i]);
		max = std::max(max, scattered_array[i]);
	}

	std::cout << "process with rank " << rank << " calculated sum = " << sum << " in " << time << " seconds\n";
	std::cout << "process with rank " << rank << " calculated min = " << min << "\n";
	std::cout << "process with rank " << rank << " calculated max = " << max << "\n\n";

	MPI_Reduce(&sum, &final_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&min, &final_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&max, &final_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		std::cout << "total sum is " << final_sum << "\n";
		std::cout << "final min is " << final_min << "\n";
		std::cout << "final max is " << final_max << "\n";
	}


	// terminate parallel execution
	MPI_Finalize();

	return 0;
}