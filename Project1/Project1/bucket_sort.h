#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <tuple>

void insertionSort(std::vector<int>& bucket) 
{
    for (int i = 1; i < bucket.size(); ++i) 
    {
        int key = bucket[i];
        int j = i - 1;
        while (j >= 0 && bucket[j] > key) 
        {
            bucket[j + 1] = bucket[j];
            j--;
        }
        bucket[j + 1] = key;
    }
}

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

std::vector<int> exchange_buckets(const std::vector<int>& send_buffer, const std::vector<int>& send_counts, const std::vector<int>& send_displs, int size) 
{
    std::vector<int> recv_counts(size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
        recv_counts.data(), 1, MPI_INT,
        MPI_COMM_WORLD);

    std::vector<int> recv_displs(size);
    int total_recv = 0;
    for (int i = 0; i < size; ++i) {
        recv_displs[i] = total_recv;
        total_recv += recv_counts[i];
    }

    std::vector<int> local_bucket(total_recv);
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_INT,
        local_bucket.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
        MPI_COMM_WORLD);

    return local_bucket;
}

std::vector<int> bucket_sort_local(const std::vector<int>& data, int num_buckets, int max_val) 
{
    auto buckets = assign_range_buckets(data, num_buckets, max_val);
    std::vector<int> sorted;
    for (auto& bucket : buckets) {
        insertionSort(bucket);
        sorted.insert(sorted.end(), bucket.begin(), bucket.end());
    }
    return sorted;
}

std::vector<int> mpi_gather(const std::vector<int>& local_data, int rank, int size) 
{
    int local_size = local_data.size();
    std::vector<int> recv_sizes(size);
    MPI_Gather(&local_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> recv_offsets(size);
    int total_size = 0;
    if (rank == 0) 
    {
        for (int i = 0; i < size; ++i) 
        {
            recv_offsets[i] = total_size;
            total_size += recv_sizes[i];
        }
    }

    std::vector<int> global_data(rank == 0 ? total_size : 0);
    MPI_Gatherv(local_data.data(), local_size, MPI_INT,
        global_data.data(), recv_sizes.data(), recv_offsets.data(), MPI_INT,
        0, MPI_COMM_WORLD);

    return global_data;
}

void MPI_BucketSort(std::vector<int>& global_data, int rank, int size) 
{
    int global_size = global_data.size();
    int local_size = global_size / size;
    std::vector<int> local_data(local_size);

    MPI_Scatter(global_data.data(), local_size, MPI_INT, local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_max = *std::max_element(local_data.begin(), local_data.end());
    int global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    auto buckets = assign_range_buckets(local_data, size, global_max);
    auto [send_buffer, send_counts, send_displs] = flatten(buckets);

    auto received_data = exchange_buckets(send_buffer, send_counts, send_displs, size);
    auto local_sorted = bucket_sort_local(received_data, size, global_max);

    global_data = mpi_gather(local_sorted, rank, size);
}