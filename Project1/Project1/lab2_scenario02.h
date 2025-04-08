#include<iostream>
#include<mpi.h>

// SCENARIO 02 - COMUNICARE BROADCAST

int main(int argc, char** argv)
{
	int rank, size;

	// parallel execution initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int broadcast_message = 77; // initial value
	if (rank == 0)
	{
		broadcast_message = 33; // broadcast value
	}
	// BROADCAST FUNCTION
	MPI_Bcast(&broadcast_message, 1, MPI_INT, 0, MPI_COMM_WORLD);
	std::cout << "process with rank " << rank << " has received message " << broadcast_message << "\n";

	// terminate parallel execution
	MPI_Finalize();

	return 0;
}