#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <tuple>

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
    for (int i = 0; i < size; ++i) {
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