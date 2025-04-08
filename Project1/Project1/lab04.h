#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

//////// For 3a

// Allocate space for a matrix with a single contiguous block
int** alloc_matrix(int n, int m)
{
	// blocuri diferite
	/*int** c = new int* [n];
	for (int i = 0; i < n; i++)
	{
		c[i] = new int[m];
	}*/
	// blocu continuu
	int** a = new int* [n];
	a[0] = new int[n * m];
	for (int i = 1; i < n; i++)
	{
		a[i] = a[i - 1] + m;
	}

	return a;
}

// Initialize a matrix with random values
void init_matrix(int n, int m, int** a)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			a[i][j] = rand() % 14;
		}
	}
}

// Free contiguous block for a matrix
void free_matrix(int** mat)
{
	if (!mat)
		return;
	delete[] mat[0];
	delete[] mat;
}

// Classic product matrix function
int** prod_matrix(int n, int m, int p, int** a, int** b)
{
	int** c = alloc_matrix(n, m);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			c[i][j] = 0;
			for (int k = 0; k < p; k++)
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	return c;
}

//////// For 3b
int MPI_Prod_matrix(int n, int** a, int** b, int** c, int root, MPI_Comm comm)
{
	// get rank and size of comm
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// alocate space for local_a and local_c
	int** local_a, ** local_c;
	local_a = alloc_matrix(n / size, n);
	local_c = alloc_matrix(n / size, n);

	// scatter a to local_a and bcast b
	MPI_Scatter(a[0], n / size * n, MPI_INT, local_a[0], n / size * n, MPI_INT, root, comm);
	MPI_Bcast(b[0], n * n, MPI_INT, root, comm);

	// calculate local_c = local_a * b
	local_c = prod_matrix(n / size, n, n, local_a, b);

	// gather local_c
	MPI_Gather(local_c[0], n / size * n, MPI_INT, c[0], n / size * n, MPI_INT, root, comm);

	// free matrix
	free_matrix(local_a);
	free_matrix(local_c);

	return MPI_SUCCESS;
}

//////// For 3c
int** trans_matrix(int n, int m, int** a)
{
	int** b = alloc_matrix(m, n);

	for (int j = 0; j < m; j++)
	{
		for (int i = 0; i < n; i++)
		{
			b[j][i] = a[i][j];
		}
	}

	return b;
}

// Same as prod_matrix but with b[j][k] instead of b[k][j]
int** pseudo_prod_matrix(int n, int m, int p, int** a, int** b)
{
	int** c = alloc_matrix(n, m);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			c[i][j] = 0;
			for (int k = 0; k < p; k++)
			{
				c[i][j] += a[i][k] * b[j][k];
			}
		}
	}

	return c;
}

int MPI_Prod_matrix_pseudo(int n, int** a, int** b, int** c, int root, MPI_Comm comm)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int** local_a, ** local_c;
	local_a = alloc_matrix(n / size, n);
	local_c = alloc_matrix(n / size, n);

	MPI_Scatter(a[0], n / size * n, MPI_INT, local_a[0], n / size * n, MPI_INT, root, comm);
	MPI_Bcast(b[0], n * n, MPI_INT, root, comm);

	local_c = pseudo_prod_matrix(n / size, n, n, local_a, trans_matrix(n, n, b)); // inmultim a cu b transpus

	MPI_Gather(local_c[0], n / size * n, MPI_INT, c[0], n / size * n, MPI_INT, root, comm);

	free_matrix(local_a);
	free_matrix(local_c);

	return MPI_SUCCESS;
}

//////// For 3d
int MPI_Prod_matrix_pseudo_row(int n, int** a, int** b, int** c, int root, MPI_Comm comm)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Datatype row;
	MPI_Type_contiguous(n, MPI_INT, &row);
	MPI_Type_commit(&row);

	int** local_a, ** local_c;
	local_a = alloc_matrix(n / size, n);
	local_c = alloc_matrix(n / size, n);

	MPI_Scatter(a[0], n / size, row, local_a[0], n / size, row, root, comm);
	MPI_Bcast(b[0], n, row, root, comm);

	MPI_Gather(local_c[0], n / size, row, c[0], n / size, row, root, comm);

	free_matrix(local_a);
	free_matrix(local_c);

	return MPI_SUCCESS;
}

int main(int argc, char** argv) {
	int size, rank;
	int n = 1'000;
	int** a, ** b, ** c;
	double time;

	srand(std::time(0));

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// 1 - Allocate memory for matrices a, b and c using alloc_matrix function
	a = alloc_matrix(n, n);
	b = alloc_matrix(n, n);
	c = alloc_matrix(n, n);

	// 2 - Initialise the matrices a and b using init_matrix function on rank 0
	if (rank == 0)
	{
		init_matrix(n, n, a);
		init_matrix(n, n, b);
	}


	// 3 - Initialize c with the result of a matrix product function
	// Count and print the time taken for each used function

	time = MPI_Wtime();

	// 3a - prod_matrix - Classic product matrix function on rank 0

	if (rank == 0)
	{
		//c = prod_matrix(n, n, n, a, b);
	}

	time = MPI_Wtime() - time;
	std::cout << "process with rank " << rank << " took " << time << " seconds\n";

	// 3b - MPI_Prod_matrix - First iteration of parallelization
	//MPI_Prod_matrix(n, a, b, c, 0, MPI_COMM_WORLD);

	// 3c - MPI_Prod_matrix_pseudo - A pseudo matrix + trans_matrix
	//MPI_Prod_matrix_pseudo(n, a, b, c, 0, MPI_COMM_WORLD);

	// 3d - MPI_Prod_matrix_pseudo_row - A variant with MPI_Datatype row
	MPI_Prod_matrix_pseudo_row(n, a, b, c, 0, MPI_COMM_WORLD);

	// 4 - Free memory for matrices a, b and c using free_matrix function
	free_matrix(a);
	free_matrix(b);
	free_matrix(c);

	MPI_Finalize();

	return 0;
}