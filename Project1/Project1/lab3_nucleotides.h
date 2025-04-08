#include<iostream>
#include<mpi.h>
#include <ctime>
#include<cstdlib>
#include<vector>
#include<string>
#include<fstream>

std::string read_text_from_file(const std::string& fileName)
{
	std::ifstream file(fileName, std::ios::ate);
	if (!file)
	{
		std::cout << "file does not exist!\n";
		return "";
	}

	std::streamsize streamSize = file.tellg();
	// returns how many characters there are til the current position
	// std::ios::ate moved us to the last character of the file; we opened the file at the last character

	file.seekg(0, std::ios::beg); // moving cursor to the beginning of the file

	std::string data(streamSize, '\n');
	file.read(&data[0], streamSize);
	return data;
}

int brute_force_search(const std::string& text, const std::string& pattern)
{
	int n = text.size();
	int m = pattern.size();
	int count = 0;
	bool is_match;

	for (int i = 0; i <= n - m; i++)
	{
		is_match = true;
		for (int j = 0; j < m; j++)
		{
			if (text[i + j] != pattern[j])
			{
				is_match = false;
				break;
			}
		}
		if (is_match)
		{
			count++;
		}
	}

	return count;
}

std::vector<int> compute_LPS(const std::string& pattern)
{
	int m = pattern.size();
	std::vector<int> lps(m, 0);
	int len = 0, i = 1;

	while (i < m)
	{
		if (pattern[i] == pattern[len])
		{
			// len +=1
			// lps[i] = len
			// i ++
			lps[i++] = ++len;
		}
		else if (len != 0)
		{
			len = lps[len - 1];
		}
		else
		{
			lps[i++] = 0;
		}
	}
	return lps;
}

int kmp_search(const std::string& text, const std::string& pattern)
{
	int n = text.size(), m = pattern.size(), count = 0;
	std::vector<int> lps = compute_LPS(pattern);
	int i = 0, j = 0;

	while (i < n)
	{
		if (text[i] == pattern[j])
		{
			i++;
			j++;
		}
		else
		{
			if (j != 0)
			{
				j = lps[j - 1];
			}
			else
			{
				i++;
			}
		}
		if (j == m)
		{
			count++;
			j = lps[j - 1];
		}
	}
	return count;
}

int main(int argc, char** argv)
{
	int rank, size;

	// parallel execution initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::string text;
	std::string pattern = "ATCG";
	double time;

	if (rank == 0)
	{
		time = MPI_Wtime();
		text = read_text_from_file("nucleotides_1000000000.txt");
		time = MPI_Wtime() - time;
		std::cout << "total reading time: " << time << "\n";
	}

	int textLength = text.size();
	MPI_Bcast(&textLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
	text.resize(textLength);

	MPI_Bcast(const_cast<char*>(text.data()), textLength, MPI_CHAR, 0, MPI_COMM_WORLD);

	int chunk_size = textLength / size;
	int start = rank * chunk_size;
	int end = (rank == size + 1) ? textLength : start + chunk_size; // checks if it's the last one

	time = MPI_Wtime();
	int count = kmp_search(text.substr(start, end - start), pattern);
	time = MPI_Wtime() - time;

	std::cout << "process with rank " << rank << " found " << count << " in " << time << " seconds\n";

	int total_count;
	MPI_Reduce(&count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		std::cout << "total number of apparitions " << total_count << "\n";
	}

	// terminate parallel execution
	MPI_Finalize();

	return 0;
}