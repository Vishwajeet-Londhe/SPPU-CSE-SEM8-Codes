#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Sequential BFS
    void sequentialBFS(int start) {

        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "\nSequential BFS: ";

        while (!q.empty()) {

            int node = q.front();
            q.pop();

            cout << node << " ";

            for (int neighbor : adj[node]) {

                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        cout << endl;
    }

    // Parallel BFS
    void parallelBFS(int start) {

        vector<bool> visited(V, false);
        vector<int> frontier, next;

        frontier.push_back(start);
        visited[start] = true;

        cout << "\nParallel BFS: ";

        while (!frontier.empty()) {

            next.clear();

            #pragma omp parallel for
            for (int i = 0; i < frontier.size(); i++) {

                int node = frontier[i];

                #pragma omp critical
                cout << node << " ";

                for (int neighbor : adj[node]) {

                    bool add = false;

                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            add = true;
                        }
                    }

                    if (add) {
                        #pragma omp critical
                        next.push_back(neighbor);
                    }
                }
            }

            frontier = next;
        }

        cout << endl;
    }
};

int main() {

    int V, E, start;

    cout << "Enter vertices and edges: ";
    cin >> V >> E;

    Graph g(V);

    cout << "Enter edges:\n";

    for (int i = 0; i < E; i++) {

        int u, v;
        cin >> u >> v;

        g.addEdge(u, v);
    }

    cout << "Enter starting vertex: ";
    cin >> start;

    double t1 = omp_get_wtime();
    g.sequentialBFS(start);
    double t2 = omp_get_wtime();

    cout << "Sequential Time: "
         << t2 - t1 << " seconds\n";

    t1 = omp_get_wtime();
    g.parallelBFS(start);
    t2 = omp_get_wtime();

    cout << "Parallel Time: "
         << t2 - t1 << " seconds\n";

    return 0;
}

/*
Example Graph:

        0
       / \
      1   2
     / \   \
    3   4   5
Example Input:
Enter vertices and edges:
6 5

Enter edges:
0 1
0 2
1 3
1 4
2 5

Enter starting vertex:
0
*/