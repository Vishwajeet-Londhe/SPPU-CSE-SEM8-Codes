// Assignment 2: Parallel Bubble Sort (Odd-Even) and Merge Sort using OpenMP
// Compile: g++ -fopenmp 2_sorting.cpp -o output
// Run:     ./output

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <omp.h>
using namespace std;

void seqBubble(int *a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                int t = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }
        }
    }
}

// Odd-Even Phase Sort: threads created once, reused across all phases
void parBubble(int *a, int n) {
    #pragma omp parallel shared(a)
    {
        for (int phase = 0; phase < n; phase++) {
            #pragma omp for
            for (int j = phase % 2; j < n - 1; j += 2) {
                if (a[j] > a[j + 1]) {
                    int t = a[j];
                    a[j] = a[j + 1];
                    a[j + 1] = t;
                }
            }
        }
    }
}

void merge(int *a, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int *L = new int[n1];
    int *R = new int[n2];
    for (int i = 0; i < n1; i++) L[i] = a[l + i];
    for (int i = 0; i < n2; i++) R[i] = a[m + 1 + i];
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) a[k++] = L[i++];
        else               a[k++] = R[j++];
    }
    while (i < n1) a[k++] = L[i++];
    while (j < n2) a[k++] = R[j++];
    delete[] L;
    delete[] R;
}

void seqMerge(int *a, int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;
    seqMerge(a, l, m);
    seqMerge(a, m + 1, r);
    merge(a, l, m, r);
}

void parMerge(int *a, int l, int r, int depth) {
    if (l >= r) return;
    int m = (l + r) / 2;
    if (depth > 0) {
        #pragma omp task
        parMerge(a, l, m, depth - 1);
        #pragma omp task
        parMerge(a, m + 1, r, depth - 1);
        #pragma omp taskwait
    } else {
        seqMerge(a, l, m);
        seqMerge(a, m + 1, r);
    }
    merge(a, l, m, r);
}

int main() {
    // Hardcoded sizes: small to large so chain graph lines cross
    int sizes[] = {1000, 5000, 10000, 30000, 50000};
    int n = 5;
    int threads = omp_get_max_threads();

    cout << "Vishwajeet Londhe BE B 41237" << endl;
    cout << "Threads: " << threads << endl << endl;
    cout << "Size\tSeqBubble\tParBubble\tSpBubble\tSeqMerge\tParMerge\tSpMerge\tEfficiency" << endl;

    ofstream csv("2_sorting.csv");
    csv << "size,seq_bubble,par_bubble,speedup_bubble,seq_merge,par_merge,speedup_merge,efficiency\n";

    for (int t = 0; t < n; t++) {
        int sz = sizes[t];

        int *base = new int[sz];
        srand(42 + t);
        for (int i = 0; i < sz; i++)
            base[i] = rand() % 100000;

        int *a1 = new int[sz];
        int *a2 = new int[sz];
        int *a3 = new int[sz];
        int *a4 = new int[sz];
        memcpy(a1, base, sz * sizeof(int));
        memcpy(a2, base, sz * sizeof(int));
        memcpy(a3, base, sz * sizeof(int));
        memcpy(a4, base, sz * sizeof(int));

        double start;

        start = omp_get_wtime();
        seqBubble(a1, sz);
        double seqBub = omp_get_wtime() - start;

        start = omp_get_wtime();
        parBubble(a2, sz);
        double parBub = omp_get_wtime() - start;

        start = omp_get_wtime();
        seqMerge(a3, 0, sz - 1);
        double seqMer = omp_get_wtime() - start;

        start = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            parMerge(a4, 0, sz - 1, 4);
        }
        double parMer = omp_get_wtime() - start;

        double spB = seqBub / parBub;
        double spM = seqMer / parMer;
        double eff = ((spB + spM) / 2.0) / threads;

        cout << sz << "\t"
             << seqBub << "\t" << parBub << "\t" << spB << "\t"
             << seqMer << "\t" << parMer << "\t" << spM << "\t" << eff << endl;

        csv << sz << ","
            << seqBub << "," << parBub << "," << spB << ","
            << seqMer << "," << parMer << "," << spM << "," << eff << "\n";

        delete[] base;
        delete[] a1;
        delete[] a2;
        delete[] a3;
        delete[] a4;
    }

    csv.close();
    cout << "\nSaved to 2_sorting.csv" << endl;
    return 0;
}