#include<iostream>
#include<mpi.h>

// SCENARIO 01 - COMUNICARE DE BAZA P2P
// procesul cu rank 0 trimite mesaj
// (numar/string) catre procesul cu rank 1

int main(int argc, char** argv)
{
	int rank, size;

	// parallel execution initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0)
	{
		const int number_message = 20;
		// tag = comunicator secret (int, penultim_
		MPI_Send(&number_message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		std::cout << "proccess with rank " << rank << " has sent message nr." << number_message << "\n";

		const char* string_message = "hi";
		MPI_Send(string_message, strlen(string_message) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
		std::cout << "proccess with rank " << rank << " has sent message : " << string_message << "\n";
	}
	else if (rank == 1)
	{
		int number_message;
		MPI_Recv(&number_message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		std::cout << "proccess with rank " << rank << " has received message nr." << number_message << "\n";

		MPI_Status status;
		MPI_Probe(0, 0, MPI_COMM_WORLD, &status);

		int msg_size;
		MPI_Get_count(&status, MPI_CHAR, &msg_size);
		std::cout << "size = " << size << "\n";

		char* string_message = new char[size];
		MPI_Recv(string_message, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		std::cout << "proccess with rank " << rank << " has received message : " << string_message << "\n";
	}


	// terminate parallel execution
	MPI_Finalize();

	return 0;
}