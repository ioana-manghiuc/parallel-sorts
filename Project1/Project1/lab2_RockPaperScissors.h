#include<iostream>
#include<mpi.h>
#include <ctime>
#include<cstdlib>

// rock paper scissors

int GenerateChoice()
{
	return rand() % 3;
}

int EstablishWinner(int choice1, int choice2)
{
	if (choice1 == choice2)
	{
		return 0;
	}
	if ((choice1 == 0 && choice2 == 1)
		|| (choice1 == 1 && choice2 == 2)
		|| (choice1 == 2 && choice2 == 0))
	{
		return 1;
	}
	return 2;
}

int main(int argc, char** argv)
{
	int rank, size;

	// parallel execution initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	srand(time(NULL) * (rank + 1));
	int choice = GenerateChoice();
	int tag = 0;

	if (rank == 0)
	{
		int other_choice;
		MPI_Send(&choice, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
		MPI_Recv(&other_choice, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		std::cout << "player 1 has chosen " << choice << "\n";
		std::cout << "player 2 has chosen " << other_choice << "\n";
		int winner = EstablishWinner(choice, other_choice);
		if (winner == 0)
		{
			std::cout << "it's a draw!\n";
		}
		else
		{
			std::cout << "player " << winner << " has won!\n";
		}
	}
	else if (rank == 1)
	{
		int other_choice;
		MPI_Send(&choice, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
		MPI_Recv(&other_choice, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}


	// terminate parallel execution
	MPI_Finalize();

	return 0;
}