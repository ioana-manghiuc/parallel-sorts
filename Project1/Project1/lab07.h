#include <iostream>
#include <vector>
#include <mpi.h>
#include <algorithm>

/*
 - numar de elemente trebuie sa fie putere a lui 2 (vezi sort visualizer online)
 tema: ITERATIVE SORTS !!!

 i&k = 0 => panta crescatoare

*/

bool isPowerOfTwo(int n)
{
    // Check if n is power of 2 --> bit shifting
    // n este putere a lui doi daca shiftat cu n-1 este 0
    return (n > 0) && (n & (n - 1)) == 0;
}

void bitonicSort(std::vector<int>& arr)
{
    int k, j, l, i, temp;

    for (k = 2; k <= arr.size(); k *= 2)
    {
        for (j = k / 2; j > 0; j /= 2)
        {
            for (i = 0; i < arr.size(); i++)
            {
                l = i ^ j;
                if (l > i && l < arr.size())
                {
                    if ((((i & k) == 0) && (arr[i] > arr[l])) ||
                        (((i & k) != 0) && (arr[i] < arr[l])))
                    {
                        temp = arr[i];
                        arr[i] = arr[l];
                        arr[l] = temp;
                    }
                }
            }
        }
    }

}


void MPI_Bitonic(std::vector<int>& global_arr, int rank, int size) {
    // Get global_size, local_size and populate local_arr
    int global_size = global_arr.size();
    int local_size = global_size / size; // => daca nr de procese nu este putere a lui 2 nu va functiona
    std::vector<int> local_arr(local_size);

    // Scatter global_arr to all processes
    MPI_Scatter(global_arr.data(), local_size, MPI_INT, local_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort local_arr
    std::sort(local_arr.begin(), local_arr.end());

    // Perform the bitonic sorting
    for (int k = 2; k <= size; k *= 2) // Use one loop for Bitonic sort step size
    {
        for (int j = k / 2; j > 0; j /= 2) // Use one loop for Iterations
        {
            // Get partner for comparison (rank ^ j)
            int partner = (rank ^ j);
            if (partner < size)
            {   // Check if partner is valid
                // Declare partner_arr
                // Exchange data with partner
                // Declare merged_arr
                // Merge local array and partner's array correctly
                // Determine sorting direction (ascending or descending)
                std::vector <int> partner_arr(local_size);
                MPI_Sendrecv(local_arr.data(), local_size, MPI_INT, partner, 0, partner_arr.data(), local_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<int> merged_arr(2 * local_size);
                std::merge(local_arr.begin(), local_arr.end(), partner_arr.begin(), partner_arr.end(), merged_arr.begin());
                bool ascending = ((rank / k) % 2 == 0);
                if (ascending)
                {
                    std::copy(merged_arr.begin(), merged_arr.begin() + local_size, local_arr.begin());
                }
                else
                {
                    std::copy(merged_arr.end() - local_size, merged_arr.end(), local_arr.begin());
                }
            }
            // Synchronize after each step
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Gather the sorted local arrays back to the root process
    MPI_Gather(local_arr.data(), local_size, MPI_INT, global_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Create a copy of the gathered array to hold the final sorted result
        // sortam direct global_arr
       //std::sort(global_arr.begin(), global_arr.end());

        // interclasare
        std::vector<int> final_arr(global_arr);
        int step = local_size;
        for (int i = 0; i < size; i++)
        {
            std::merge(final_arr.begin(), final_arr.begin() + (i * step),
                global_arr.begin() + (i * step), global_arr.begin() + ((i + 1) * step), final_arr.begin());
        }

        global_arr = final_arr;
        // Iterate over each chunk of the array and merge them one by one

        // Merge the current chunk with the final sorted array

        // Assign the merged result back to global_arr
    }
}


int main(int argc, char** argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int N = 1'048'576;
    std::vector<int> global_arr(N);

    if (isPowerOfTwo(N) != 1 || isPowerOfTwo(size) == false)
    {
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    if (rank == 0)
    {
        srand(time(NULL) * 0 + 1024);
        for (int i = 0; i < N; i++)
        {
            global_arr[i] = rand();
        }
    }

    double time = MPI_Wtime();

    /*if (rank == 0)
    {
        bitonicSort(global_arr);
    }*/

    MPI_Bitonic(global_arr, rank, size);
    time = MPI_Wtime() - time;

    //  bitonicSort(global_arr

    if (rank == 0)
    {
        std::cout << "sorting took " << time << " seconds.\n";
        std::cout << std::is_sorted(global_arr.begin(), global_arr.end());
    }

    MPI_Finalize();

    return 0;
}