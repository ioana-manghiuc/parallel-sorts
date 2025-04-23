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

std::vector<int> collate_arrays(std::vector<int>&& a, std::vector<int>&& b)
{
	std::vector<int> result;
	result.reserve(a.size() + b.size());

	size_t i = 0, j = 0;
	while (i < a.size() && j < b.size())
	{
		if (a[i] < b[j])
			result.push_back(std::move(a[i++]));
		else
			result.push_back(std::move(b[j++]));
	}

	while (i < a.size()) result.push_back(std::move(a[i++]));
	while (j < b.size()) result.push_back(std::move(b[j++]));

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

		std::vector<std::vector<int>> chunks(size);
		for (int i = 0; i < size; ++i)
		{
			chunks[i] = std::vector<int>(gathered_data.begin() + i * local_size,
				gathered_data.begin() + (i + 1) * local_size);
		}

		while (chunks.size() > 1)
		{
			std::vector<std::vector<int>> new_chunks;
			for (size_t i = 0; i < chunks.size(); i += 2)
			{
				if (i + 1 < chunks.size())
				{
					new_chunks.push_back(collate_arrays(std::move(chunks[i]), std::move(chunks[i + 1])));
				}
				else
				{
					new_chunks.push_back(chunks[i]);
				}
			}
			chunks = std::move(new_chunks);
		}

		global_data = std::move(chunks[0]);
		computation_time += MPI_Wtime() - start_time;
	}

}