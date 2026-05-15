// Parallel DFS using OpenMP
// Compile: g++ -fopenmp 1_dfs.cpp -o output
// Run:     ./output

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
using namespace std;

int V, **adj;

void initGraph(int v) {
    V = v;
    adj = new int*[V];
    for(int i=0;i<V;i++)
        adj[i] = new int[V]();
}

void freeGraph() {
    for(int i=0;i<V;i++)
        delete[] adj[i];
    delete[] adj;
}

void addEdge(int u,int v) {
    adj[u][v] = 1;
    adj[v][u] = 1;
}

void genEdges(int m) {
    srand(42);
    int e = 0;

    while(e < m) {
        int u = rand()%V;
        int v = rand()%V;

        if(u != v && !adj[u][v]) {
            addEdge(u,v);
            e++;
        }
    }
}

void seqDFS(int u,int vis[]) {
    vis[u] = 1;

    for(int i=0;i<V;i++)
        if(adj[u][i] && !vis[i])
            seqDFS(i,vis);
}

void parDFS(int u,int vis[]) {

    for(int i=0;i<V;i++) {

        if(adj[u][i] && !vis[i]) {

            int go = 0;

            #pragma omp critical
            {
                if(!vis[i]) {
                    vis[i] = 1;
                    go = 1;
                }
            }

            if(go) {
                #pragma omp task
                parDFS(i,vis);
            }
        }
    }

    #pragma omp taskwait
}

int main() {

    int sizes[] = {100,500,2000,5000,10000};
    int threads = omp_get_max_threads();

    cout << "Vishwajeet Londhe BE B 41237" << endl;
    cout<<"Threads: "<<threads<<endl<<endl;
    cout<<"Size\tSeqDFS\tParDFS\tSpeedup\tEfficiency"<<endl;

    ofstream csv("dfs.csv");
    csv<<"size,seqdfs,pardfs,speedup,efficiency\n";

    for(int t=0;t<5;t++) {

        initGraph(sizes[t]);

        int edges = sizes[t] * 3;
        genEdges(edges);

        double start;

        int *v1 = new int[sizes[t]]();

        start = omp_get_wtime();
        seqDFS(0,v1);
        double seq = omp_get_wtime() - start;

        delete[] v1;

        int *v2 = new int[sizes[t]]();
        v2[0] = 1;

        start = omp_get_wtime();

        #pragma omp parallel
        {
            #pragma omp single
            parDFS(0,v2);
        }

        double par = omp_get_wtime() - start;

        delete[] v2;

        double speedup = seq/par;
        double efficiency = speedup/threads;

        cout<<sizes[t]<<"\t"<<seq<<"\t"<<par<<"\t"
            <<speedup<<"\t"<<efficiency<<endl;

        csv<<sizes[t]<<","<<seq<<","<<par<<","
           <<speedup<<","<<efficiency<<"\n";

        freeGraph();
    }

    csv.close();

    return 0;
}