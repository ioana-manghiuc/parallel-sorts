#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>

void DirectSort(std::vector<int>& data)
{
	for (int j = 0; j < data.size() - 1; ++j)
	{
		int current_min = j;
		for (int i = j + 1; i < data.size(); ++i)
		{
			if (data[i] < data[current_min])
				current_min = i;
		}
		if (current_min != j)
			std::swap(data[j], data[current_min]);
	}
}

std::vector<int> collate_arrays(const std::vector<int>& a, const std::vector<int>& b)
{
	std::vector<int> result;
	result.reserve(a.size() + b.size());

	size_t i = 0, j = 0;
	while (i < a.size() && j < b.size())
	{
		if (a[i] < b[j])
			result.push_back(a[i++]);
		else
			result.push_back(b[j++]);
	}
	while (i < a.size()) result.push_back(a[i++]);
	while (j < b.size()) result.push_back(b[j++]);

	return result;
}

void MPI_DirectSort(std::vector<int>& global_data, int rank, int size, double& computation_time, double& communication_time)
{
	double start_time = 0.0;
	int global_size = global_data.size();
	int local_size = global_size / size;

	std::vector<int> local_data(local_size);

	start_time = MPI_Wtime();
	MPI_Scatter(global_data.data(), local_size, MPI_INT,
		local_data.data(), local_size, MPI_INT,
		0, MPI_COMM_WORLD);
	communication_time += MPI_Wtime() - start_time;

	start_time = MPI_Wtime();
	DirectSort(local_data);
	computation_time += MPI_Wtime() - start_time;

	std::vector<int> gathered_data;
	if (rank == 0) gathered_data.resize(global_size);

	start_time = MPI_Wtime();
	MPI_Gather(local_data.data(), local_size, MPI_INT,
		rank == 0 ? gathered_data.data() : nullptr,
		local_size, MPI_INT, 0, MPI_COMM_WORLD);
	communication_time += MPI_Wtime() - start_time;

	if (rank == 0)
	{
		start_time = MPI_Wtime();

		global_data.assign(gathered_data.begin(), gathered_data.begin() + local_size);

		for (int i = 1; i < size; ++i)
		{
			std::vector<int> temp(gathered_data.begin() + i * local_size,
				gathered_data.begin() + (i + 1) * local_size);
			global_data = collate_arrays(global_data, temp);
		}

		computation_time += MPI_Wtime() - start_time;
	}
}