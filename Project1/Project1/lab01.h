//#include<iostream>
//#include<mpi.h>
//
// 
// // PART 1
//int main(int argc, char** argv)
//{
//	int rank, size;
//	// parallel execution initialization
//	MPI_Init(&argc, &argv);
//
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//	std::cout << "hello from process of rank " << rank << " of " << size << "\n";
//	std::cout.flush();
//
//	// stop - artificial barrier for synchronization
//	MPI_Barrier(MPI_COMM_WORLD);
//
//	if (rank == 0)
//	{
//		std::cout << "all processes halted.\n";
//	}
//
//	std::cout << "hello from process of rank " << rank << " of " << size << "\n";
//
//	// terminate parallel execution
//	MPI_Finalize();
//
//	return 0;
//}

// // PART 2

//#include<iostream>
//#include<mpi.h>
//
//// speedup = sequential_time / parallel_time
//
//bool IsPrimeNumber(long number)
//{
//	if (number == 1)
//		return 0;
//	if (number == 2 || number == 3)
//		return 1;
//	if (number % 2 == 0)
//		return 0;
//	for (int divided = 5; divided < std::sqrt(number); divided += 2)
//	{
//		if (number % divided == 0)
//		{
//			return 0;
//		}
//	}
//	return 1;
//}
//
//int main(int argc, char** argv)
//{
//	// se considera un numar n foarte mare
//	// determinam cate numere prime exista in intervalul [1, n];
//
//	int rank, size, count = 0, total_count = 0;
//	int n = 1'000'000;
//
//	// parallel execution initialization
//	MPI_Init(&argc, &argv);
//
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//	int start = rank;
//	int end = n;
//
//	double time = MPI_Wtime();
//
//	for (int i = start; i < end; i += size)
//	{
//		if (IsPrimeNumber(i))
//		{
//			count++;
//		}
//	}
//
//	time = MPI_Wtime() - time;
//
//	std::cout << "processor of rank " << rank << " found " << count << " prime numbers in " << time << " seconds\n";
//	std::cout.flush();
//
//	MPI_Reduce(&count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//
//	if (rank == 0)
//	{
//		std::cout << total_count << " prime numbers found.\n";
//	}
//
//	// terminate parallel execution
//	MPI_Finalize();
//
//	return 0;
//}