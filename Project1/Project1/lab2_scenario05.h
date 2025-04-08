#include<iostream>
#include<mpi.h>
#include<vector>

// SCENARIO 05 - COMUNICARE CIRCULARA / INEL

int main(int argc, char** argv)
{
	int rank, size;

	// parallel execution initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int left, right;
	int ibuff, obuff, sum = 0;
	int tag = 1;

	right = rank + 1; if (right == size) right = 0;
	left = rank - 1; if (left == -1) left = size - 1;
	obuff = rank;

	for (int i = 0; i < size; i++)
	{
		// send to right, receive from left
		MPI_Send(&obuff, 1, MPI_INT, right, tag, MPI_COMM_WORLD);
		MPI_Recv(&ibuff, 1, MPI_INT, left, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		sum += ibuff;
		obuff = ibuff;

	}

	std::cout << "process with rank " << rank << " has sum = " << sum << "\n";

	// terminate parallel execution
	MPI_Finalize();

	return 0;
}
