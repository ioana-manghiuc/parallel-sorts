#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <tuple>
#include <cstdlib>
#include <cmath>
#include<fstream>

/* DIRECT SORT */

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

/* BUCKET SORT */

std::vector<std::vector<int>> assign_range_buckets(const std::vector<int>& data, int num_buckets, int max_val)
{
    std::vector<std::vector<int>> buckets(num_buckets);
    int bucket_width = (max_val + 1 + num_buckets - 1) / num_buckets;

    for (int value : data)
    {
        int index = std::min(value / bucket_width, num_buckets - 1);
        buckets[index].push_back(value);
    }
    return buckets;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> flatten(const std::vector<std::vector<int>>& buckets) {
    std::vector<int> flat_buffer;
    std::vector<int> elements_per_bucket(buckets.size());
    std::vector<int> bucket_start_indexes(buckets.size());

    int offset = 0;
    for (int i = 0; i < buckets.size(); ++i) {
        elements_per_bucket[i] = buckets[i].size();
        bucket_start_indexes[i] = offset;
        flat_buffer.insert(flat_buffer.end(), buckets[i].begin(), buckets[i].end());
        offset += elements_per_bucket[i];
    }
    return { flat_buffer, elements_per_bucket, bucket_start_indexes };
}

std::vector<int> exchange_buckets(const std::vector<int>& send_buffer, const std::vector<int>& send_counts, const std::vector<int>& send_displs, int size,
    double& computation_time, double& communication_time)
{
    double start_time = 0.0;

    start_time = MPI_Wtime();
    std::vector<int> recv_counts(size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
        recv_counts.data(), 1, MPI_INT,
        MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - start_time;

    start_time = MPI_Wtime();
    std::vector<int> recv_displs(size);
    int total_recv = 0;
    for (int i = 0; i < size; ++i)
    {
        recv_displs[i] = total_recv;
        total_recv += recv_counts[i];
    }
    computation_time += MPI_Wtime() - start_time;

    start_time = MPI_Wtime();
    std::vector<int> local_bucket(total_recv);
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_INT,
        local_bucket.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
        MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - start_time;

    return local_bucket;
}

std::vector<int> mpi_gather(const std::vector<int>& local_data, int rank, int size,
    double& computation_time, double& communication_time)
{
    double start_time = 0.0;
    int local_size = local_data.size();
    std::vector<int> recv_sizes(size);

    start_time = MPI_Wtime();
    MPI_Gather(&local_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - start_time;

    std::vector<int> recv_offsets(size);
    int total_size = 0;
    if (rank == 0)
    {
        start_time = MPI_Wtime();
        for (int i = 0; i < size; ++i)
        {
            recv_offsets[i] = total_size;
            total_size += recv_sizes[i];
        }
        computation_time += MPI_Wtime() - start_time;
    }

    std::vector<int> global_data(rank == 0 ? total_size : 0);

    start_time = MPI_Wtime();
    MPI_Gatherv(local_data.data(), local_size, MPI_INT,
        global_data.data(), recv_sizes.data(), recv_offsets.data(), MPI_INT,
        0, MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - start_time;

    return global_data;
}

void MPI_BucketSort(std::vector<int>& global_data, int rank, int size, double& computation_time, double& communication_time)
{
    double start_time = 0.0;
    int global_size = global_data.size();
    int local_size = global_size / size;
    std::vector<int> local_data(local_size);

    start_time = MPI_Wtime();
    MPI_Scatter(global_data.data(), local_size, MPI_INT, local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - start_time;

    start_time = MPI_Wtime();
    int local_max = *std::max_element(local_data.begin(), local_data.end());
    computation_time += MPI_Wtime() - start_time;

    int global_max;
    start_time = MPI_Wtime();
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - start_time;

    start_time = MPI_Wtime();
    auto buckets = assign_range_buckets(local_data, size, global_max);
    auto [send_buffer, send_counts, send_displs] = flatten(buckets);
    computation_time += MPI_Wtime() - start_time;

    auto received_data = exchange_buckets(send_buffer, send_counts, send_displs, size, computation_time, communication_time);

    start_time = MPI_Wtime();
    std::sort(received_data.begin(), received_data.end());
    computation_time += MPI_Wtime() - start_time;

    global_data = mpi_gather(received_data, rank, size, computation_time, communication_time);
}

/* ODD-EVEN SORT */

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

/* RANKING SORT */

void MPI_RankingSort(std::vector<int>& data, int rank, int size, double& computation_time, double& communication_time)
{
	double start_time = 0.0;
	int n = data.size();
	int local_n = n / size;

	std::vector<int> full_data(n);
	std::vector<int> ranking(local_n);
	std::vector<int> overall_ranking(n);

	start_time = MPI_Wtime();
	MPI_Bcast(data.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
	communication_time += MPI_Wtime() - start_time;

	full_data = data;

	start_time = MPI_Wtime();
	for (int i = 0; i < local_n; ++i)
	{
		int global_i = i + rank * local_n;
		ranking[i] = 0;
		for (int j = 0; j < n; ++j)
		{
			if (full_data[j] < full_data[global_i] ||
				(full_data[j] == full_data[global_i] && j < global_i))
			{
				ranking[i]++;
			}
		}
	}
	computation_time += MPI_Wtime() - start_time;

	start_time = MPI_Wtime();
	MPI_Gather(ranking.data(), local_n, MPI_INT, overall_ranking.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
	communication_time += MPI_Wtime() - start_time;

	if (rank == 0)
	{
		start_time = MPI_Wtime();
		std::vector<int> sorted(n);
		for (int i = 0; i < n; ++i)
		{
			sorted[overall_ranking[i]] = full_data[i];
		}
		data = sorted;
		computation_time += MPI_Wtime() - start_time;
	}
}

/* SHELL SORT */

bool compareAndMerge(int rank, int partner, std::vector<int>& local_data, int local_n, double& computation_time, double& communication_time)
{
	double start_time = 0.0;
	std::vector<int> recv_data(local_n);

	start_time = MPI_Wtime();
	MPI_Sendrecv(local_data.data(), local_n, MPI_INT, partner, 0,
		recv_data.data(), local_n, MPI_INT, partner, 0,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	communication_time += MPI_Wtime() - start_time;

	start_time = MPI_Wtime();
	std::vector<int> merged(local_n * 2);
	std::merge(local_data.begin(), local_data.end(), recv_data.begin(), recv_data.end(), merged.begin());

	std::vector<int> original = local_data;

	if (rank < partner)
	{
		std::copy(merged.begin(), merged.begin() + local_n, local_data.begin());
	}
	else
	{
		std::copy(merged.end() - local_n, merged.end(), local_data.begin());
	}
	computation_time += MPI_Wtime() - start_time;

	return local_data != original;
}

void MPI_ShellSort(std::vector<int>& data, int rank, int size, double& computation_time, double& communication_time)
{
	double start_time = 0.0;
	int n = data.size();
	int local_n = n / size;

	std::vector<int> local_data(local_n);
	start_time = MPI_Wtime();
	MPI_Scatter(data.data(), local_n, MPI_INT, local_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
	communication_time += MPI_Wtime() - start_time;

	start_time = MPI_Wtime();
	std::sort(local_data.begin(), local_data.end());
	computation_time += MPI_Wtime() - start_time;

	int max_log = static_cast<int>(std::log2(size));

	for (int l = 0; l <= max_log; ++l) {
		int stride = 1 << (max_log - l);
		int group_id = rank / stride;

		int partner = (group_id % 2 == 0) ? rank + stride : rank - stride;

		if (partner >= 0 && partner < size)
		{
			compareAndMerge(rank, partner, local_data, local_n, computation_time, communication_time);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	int is_sorted = 0;
	while (!is_sorted)
	{
		bool changed = false;

		if ((rank % 2 == 0) && (rank + 1 < size))
		{
			changed = changed || compareAndMerge(rank, rank + 1, local_data, local_n, computation_time, communication_time);
		}
		else if ((rank % 2 == 1) && (rank - 1 >= 0))
		{
			changed = changed || compareAndMerge(rank, rank - 1, local_data, local_n, computation_time, communication_time);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		if ((rank % 2 == 1) && (rank + 1 < size))
		{
			changed = changed || compareAndMerge(rank, rank + 1, local_data, local_n, computation_time, communication_time);
		}
		else if ((rank % 2 == 0) && (rank - 1 >= 0))
		{
			changed = changed || compareAndMerge(rank, rank - 1, local_data, local_n, computation_time, communication_time);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		int local_sorted = changed ? 0 : 1;
		start_time = MPI_Wtime();
		MPI_Allreduce(&local_sorted, &is_sorted, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		communication_time += MPI_Wtime() - start_time;
	}

	start_time = MPI_Wtime();
	MPI_Gather(local_data.data(), local_n, MPI_INT, data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
	communication_time += MPI_Wtime() - start_time;
}

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n = 1'000'000;
	std::ifstream data_file("data_1mil.txt", std::ifstream::in);

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

	MPI_Barrier(MPI_COMM_WORLD);
	double computation_time = 0.0, communication_time = 0.0;
	double execution_time = MPI_Wtime();

	MPI_OddEvenSort(data, rank, size, computation_time, communication_time);
	MPI_Barrier(MPI_COMM_WORLD);
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