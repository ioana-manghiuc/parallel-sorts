#include<iostream>
#include<mpi.h>
#include<vector>

// SCENARIO 03 - COMUNICARE DISTRIBUITA

int main(int argc, char** argv)
{
	int rank, size;

	// parallel execution initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::vector<int> send_buffer;
	constexpr int elements_per_process = 1;
	if (rank == 0)
	{
		send_buffer = { 7,1,2,3,5,6,7,8 };
	}

	std::vector<int> recv_buffer(elements_per_process);
	MPI_Scatter(send_buffer.data(), elements_per_process, MPI_INT, recv_buffer.data(), elements_per_process, MPI_INT, 0, MPI_COMM_WORLD);
	std::cout << "process with rank " << rank << " has: ";
	for (auto element : recv_buffer)
	{
		std::cout << element << " ";
	}

	// terminate parallel execution
	MPI_Finalize();

	return 0;
}