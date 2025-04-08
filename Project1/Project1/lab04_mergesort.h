#include <iostream>
#include <vector>
#include <mpi.h>

std::vector<double> merge_arrays(
	const std::vector<double>& a, const std::vector<double>& b)
{
	std::vector<double> c(a.size() + b.size());
	size_t i = 0, j = 0, k = 0;
	while (i < a.size() && j < b.size())
	{
		if (a[i] <= b[j])
		{
			c[k++] = a[i++];
		}
		else
		{
			c[k++] = b[j++];
		}
	}
	while (i < a.size()) c[k++] = a[i++];
	while (j < b.size()) c[k++] = b[j++];
	return c;
}


void merge_sort(std::vector<double>& arr)
{
	if (arr.size() <= 1) return;
	if (arr.size() == 2)
	{
		if (arr[0] > arr[1]) std::swap(arr[0], arr[1]);
		return;
	}
	size_t mid = arr.size() / 2;
	std::vector<double> left(arr.begin(), arr.begin() + mid);
	std::vector<double> right(arr.begin() + mid, arr.end());
	merge_sort(left);
	merge_sort(right);
	arr = merge_arrays(left, right);
}
int main(int argc, char** argv)
{
	int rank, size;
	int n = 1'000'000;
	std::vector<double> array;
	std::vector<double> local_array;
	double start_time, end_time;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int local_n = n / size;
	local_array.resize(local_n);

	if (rank == 0)
	{
		array.resize(n);
		std::srand(std::time(NULL) + rank);
		for (double& val : array)
		{
			val = (static_cast<double>(std::rand()) / RAND_MAX) * 10.0;
		}
	}

	start_time = MPI_Wtime();
	MPI_Scatter(array.data(), local_n, MPI_DOUBLE, local_array.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	merge_sort(local_array);


	if (rank == 0)
	{
		for (int i = 1; i < size; i++)
		{
			std::vector<double> temp(array.begin() + i * local_n, array.begin() + (i + 1) * local_n);
			array = merge_arrays(array, temp);
		}
		end_time = MPI_Wtime();
		std::cout << "\nExecution Time: " << (end_time - start_time) << " seconds" << std::endl;

		//write_in_file(array); // se poate crea optional si o functie de afisare in fisier
		//end_time = MPI_Wtime();
		//std::cout << "\nExecution Time: " << (end_time - start_time) << " seconds" << std::endl;
	}

	MPI_Finalize();

	return 0;

}