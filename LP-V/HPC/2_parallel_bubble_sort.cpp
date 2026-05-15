#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Sequential Bubble Sort
void bubbleSortSeq(vector<int>& arr) {

    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {

        for (int j = 0; j < n - i - 1; j++) {

            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort
void bubbleSortParallel(vector<int>& arr) {

    int n = arr.size();

    for (int i = 0; i < n; i++) {

        // Even Phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {

            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }

        // Odd Phase
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {

            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {

    int n;

    cout << "Enter size of array: ";
    cin >> n;

    vector<int> arr(n), temp;

    cout << "Enter array elements:\n";

    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    double start, end;

    // Sequential Bubble Sort
    temp = arr;

    start = omp_get_wtime();

    bubbleSortSeq(temp);

    end = omp_get_wtime();

    cout << "\nSequential Bubble Sort Time: "
         << end - start << " sec\n";

    cout << "Sorted Array: ";

    for (int x : temp) {
        cout << x << " ";
    }

    cout << endl;

    // Parallel Bubble Sort
    temp = arr;

    start = omp_get_wtime();

    bubbleSortParallel(temp);

    end = omp_get_wtime();

    cout << "\nParallel Bubble Sort Time: "
         << end - start << " sec\n";

    cout << "Sorted Array: ";

    for (int x : temp) {
        cout << x << " ";
    }

    cout << endl;

    return 0;
}

//Enter size of array:10
//Enter array elements:34 16 23 646 93 78 89 78 88 89
// openmp command= -fopenmp