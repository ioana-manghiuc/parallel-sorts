#include<iostream>
#include<mpi.h>
#include<vector>

// SCENARIO 04 - COMUNICARE POINT TO POINT

int main(int argc, char** argv)
{
	int rank, size;

	// parallel execution initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 2)
	{
		std::cerr << "we need two MPI processes\n";
		MPI_Finalize();
		return 1;
	}

	int in_msg, out_msg;
	int tag = 1;
	int destination = (rank == 0) ? 1 : 0;
	int source = destination;
	out_msg = (rank == 0) ? 1234 : 5678;

	std::cout << "process with rank " << rank << " sends " << out_msg << " to " << destination << "\n";

	MPI_Sendrecv(&out_msg, 1, MPI_INT, destination, tag, &in_msg, 1, MPI_INT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	std::cout << "process with rank " << rank << " received " << in_msg << "\n";

	// terminate parallel execution
	MPI_Finalize();

	return 0;
}