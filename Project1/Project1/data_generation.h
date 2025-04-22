#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <random>

int main() {
    const int N = 1'000'000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(data.begin(), data.end(), rng);

    std::ofstream out("data_1mil.txt");
    if (!out)
    {
        std::cerr << "Could not open file!\n";
        return 1;
    }

    for (int num : data)
    {
        out << num << '\n';
    }

    out.close();
    return 0;
}