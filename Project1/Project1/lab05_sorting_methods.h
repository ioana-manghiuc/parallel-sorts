#pragma once
#include <vector>
#include <functional>
#include <mpi.h>

std::vector<int> collate_arrays(const std::vector<int>& a, const std::vector<int>& b)
{
	std::vector<int> c(a.size() + b.size());
	size_t i = 0, j = 0, k = 0;
	while (i < a.size() && j < b.size())
	{
		if (a[i] <= b[j])
		{
			c[k++] = a[i++];
		}
		else
		{
			c[k++] = b[j++];
		}
	}
	while (i < a.size()) c[k++] = a[i++];
	while (j < b.size()) c[k++] = b[j++];
	return c;
}

void merge_sort(std::vector<int>& arr)
{
	if (arr.size() <= 1) return;
	if (arr.size() == 2)
	{
		if (arr[0] > arr[1]) std::swap(arr[0], arr[1]);
		return;
	}
	size_t mid = arr.size() / 2;
	std::vector<int> left(arr.begin(), arr.begin() + mid);
	std::vector<int> right(arr.begin() + mid, arr.end());
	merge_sort(left);
	merge_sort(right);
	arr = collate_arrays(left, right);
}

void bubble_sort(std::vector<int>& arr) {
	int n = arr.size();
	bool swapped;

	for (int i = 0; i < n - 1; i++) {
		swapped = false;
		for (int j = 0; j < n - i - 1; j++) {
			if (arr[j] > arr[j + 1]) {
				std::swap(arr[j], arr[j + 1]);
				swapped = true;
			}
		}

		// If no two elements were swapped, then break
		if (!swapped)
			break;
	}
}

void selection_sort(std::vector<int>& arr) {
	int n = arr.size();

	for (int i = 0; i < n - 1; ++i) {

		// Assume the current position holds
		// the minimum element
		int min_idx = i;

		// Iterate through the unsorted portion
		// to find the actual minimum
		for (int j = i + 1; j < n; ++j) {
			if (arr[j] < arr[min_idx]) {

				// Update min_idx if a smaller
				// element is found
				min_idx = j;
			}
		}

		// Move minimum element to its
		// correct position
		std::swap(arr[i], arr[min_idx]);
	}
}

void insertion_sort(std::vector<int>& vec)
{
	for (int i = 1; i < vec.size(); ++i) {
		int key = vec[i];
		int j = i - 1;

		/* Move elements of arr[0..i-1], that are
		   greater than key, to one position ahead
		   of their current position */
		while (j >= 0 && vec[j] > key) {
			vec[j + 1] = vec[j];
			j = j - 1;
		}
		vec[j + 1] = key;
	}
}

int partition(std::vector<int>& arr, int low, int high) {

	// Choose the pivot
	int pivot = arr[high];

	// Index of smaller element and indicates 
	// the right position of pivot found so far
	int i = low - 1;

	// Traverse arr[low..high] and move all smaller
	// elements on left side. Elements from low to 
	// i are smaller after every iteration
	for (int j = low; j <= high - 1; j++) {
		if (arr[j] < pivot) {
			i++;
			std::swap(arr[i], arr[j]);
		}
	}

	// Move pivot after smaller elements and
	// return its position
	std::swap(arr[i + 1], arr[high]);
	return i + 1;
}

// The QuickSort function implementation
void quickSort(std::vector<int>& arr, int low, int high) {

	if (low < high) {

		// pi is the partition return index of pivot
		int pi = partition(arr, low, high);

		// Recursion calls for smaller elements
		// and greater or equals elements
		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
}

void quick_sort(std::vector<int>& arr)
{
	quickSort(arr, 0, arr.size() - 1);
}

void heapify(std::vector<int>& arr, int n, int i) {

	// Initialize largest as root
	int largest = i;

	// left index = 2*i + 1
	int l = 2 * i + 1;

	// right index = 2*i + 2
	int r = 2 * i + 2;

	// If left child is larger than root
	if (l < n && arr[l] > arr[largest])
		largest = l;

	// If right child is larger than largest so far
	if (r < n && arr[r] > arr[largest])
		largest = r;

	// If largest is not root
	if (largest != i) {
		std::swap(arr[i], arr[largest]);

		// Recursively heapify the affected sub-tree
		heapify(arr, n, largest);
	}
}

// Main function to do heap sort
void heap_sort(std::vector<int>& arr) {
	int n = arr.size();

	// Build heap (rearrange vector)
	for (int i = n / 2 - 1; i >= 0; i--)
		heapify(arr, n, i);

	// One by one extract an element from heap
	for (int i = n - 1; i > 0; i--) {

		// Move current root to end
		std::swap(arr[0], arr[i]);

		// Call max heapify on the reduced heap
		heapify(arr, i, 0);
	}
}

void MPI_Sort(std::vector<int> global_data, int rank, int size, std::function<void(std::vector<int>&)> function_sort)
{
	// global_size and local_size
	int global_size = global_data.size();
	int local_size = global_size / size;

	// vector local_data with local_size
	std::vector<int> local_data(local_size);

	// scatter values from global_data into local_data
	MPI_Scatter(global_data.data(), local_size, MPI_INT, local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

	// call sort method for local_data
	function_sort(local_data);

	// gather_data with all local_data values
	std::vector<int> gathered_data(global_size);
	MPI_Gather(local_data.data(), local_size, MPI_INT, gathered_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

	// process with rank 0 collates arrays (interclasare)
	if (rank == 0)
	{
		std::vector<int> sorted_data(local_size);

		for (int i = 0; i < size; i++)
		{
			int start_index = i * local_size;
			std::vector<int> next_chunk(gathered_data.begin() + start_index, gathered_data.begin() + start_index + local_size);
			sorted_data = collate_arrays(sorted_data, next_chunk);
		}
		global_data = sorted_data;
	}
}