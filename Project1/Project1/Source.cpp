#include<iostream>
#include<mpi.h>

#include<fstream>
#include<random>
#include<vector>

int main()
{
	srand(time(NULL));
	int N = 10'000'000;
	int n = 1'000;
	std::ofstream small_data_file("small_data.txt", std::ofstream::out);
	std::ifstream read_small_data("small_data.txt", std::ifstream::in);

	std::random_device RD; 
	std::mt19937 engine(RD()); 
	std::uniform_int_distribution<> distr(0, n - 1);

	for (size_t i = 0; i < n; i++)
	{
		small_data_file << distr(engine) << " ";
	}
	std::vector<int>small_data(n);
	for (size_t i = 0; i < n; i++)
	{
		read_small_data >> small_data[i];
	}

	for (size_t i = 0; i < small_data.size(); i++)
	{
		std::cout << small_data[i] << "\n";
	}

	small_data_file.close();
	read_small_data.close();
	return 0;
}