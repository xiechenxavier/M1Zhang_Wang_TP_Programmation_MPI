#include <math.h>
#include <mpi.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

bool verboseMode = true;

// lit une instance donnée par le nom du fichier  fileName, et crÃ©e le sac Ã
// dos correspondant dans les tableaux donnÃ©s par pointeur
void readInstance(const string& fileName, vector<int>& weights,
                  vector<int>& values, int& knapsackBound, int& nbItems) {
  ifstream infile;
  infile.exceptions(ifstream::failbit | ifstream::badbit);
  infile.open(fileName.c_str());

  infile >> nbItems;

  weights.resize(nbItems);
  for (int i = 0; i < nbItems; i++) infile >> weights[i];

  values.resize(nbItems);
  for (int i = 0; i < nbItems; i++) infile >> values[i];

  infile >> knapsackBound;
  infile.close();
  /*
          cout << "print knapsack: " << endl;
          cout << "maximal weight: " << knapsackBound << endl;
          for (int i = 0; i < nbItems; i++) cout << "item: " << i << " weight: "
     << weights[i] << " value: " << values[i] << endl;
  */
}

void printKnapsackSolution(vector<bool>& solution) {
  cout << "knapsack composition  : ";
  for (std::vector<bool>::iterator it = solution.begin(); it != solution.end();
       ++it)
    std::cout << ' ' << *it;
  cout << endl;
}

void DistributedDP(vector<int>& weights, vector<int>& values, int knapsackBound,
                   int nbItems, int& costSolution, vector<bool>& solution,
                   unsigned int** matrixDP, int rankID, int nbprocs) {
  MPI_Barrier(MPI_COMM_WORLD);
  int ceil_charge_unit = ceil((double)knapsackBound / (double)nbprocs);
  int local_matrix[nbItems][ceil_charge_unit];
  int num;
  int m;
  MPI_Request send_request, recv_request;
  int nb_nums = min(weights[1], ceil_charge_unit);
  MPI_Status status;
  for (int k = 0; k < ceil_charge_unit; k++) {
      local_matrix[0][k] = matrixDP[0][rankID*ceil_charge_unit + k + 1];
  }
  for (int j = 0; j < nb_nums; j++) {
    if (rankID != nbprocs - 1)
      MPI_Isend(&local_matrix[0][ceil_charge_unit-1-j], 1, MPI_INT, rankID + 1, 1, MPI_COMM_WORLD, &send_request);
  }
  for (int i = 1; i < nbItems; i++) {
    for (int j = ceil_charge_unit - 1; j >= 0; j--) {
      if (j + 1 <= nb_nums && rankID != 0) {
        MPI_Irecv(&num, 1, MPI_INT, rankID - 1, 1, MPI_COMM_WORLD, &recv_request);
        MPI_Wait(&recv_request, &status);
      } else if (rankID == 0) {
        num = j - weights[i] < 0 ? 0 : local_matrix[i-1][j-weights[i]];
      } else {
        num = local_matrix[i-1][j-weights[i]];
      }
      m = rankID * ceil_charge_unit + (j + 1);
      if (weights[i] <= m) {
        local_matrix[i][j] = max(
            values[i] + num, local_matrix[i - 1][j]);
      } else {
        local_matrix[i][j] = local_matrix[i - 1][j];
      }
    }
    if (i != nbItems - 1) nb_nums = min(weights[i+1], ceil_charge_unit);
    if (rankID != nbprocs - 1) {
      for (int k = 0; k < nb_nums; k++) { 
        MPI_Bsend(&local_matrix[i][ceil_charge_unit - k - 1], 1, MPI_INT, rankID + 1, 1, MPI_COMM_WORLD);
        MPI_Wait(&send_request, &status);
      }
    }
    int vec_buffer[nbprocs * ceil_charge_unit];
    MPI_Allgather(local_matrix[i], ceil_charge_unit, MPI_INT, vec_buffer, ceil_charge_unit,
                  MPI_INT, MPI_COMM_WORLD);
    for (int k = 1; k < knapsackBound + 1; k++) {
      matrixDP[i][k] = vec_buffer[k-1];
    }
  }
  costSolution = matrixDP[nbItems - 1][knapsackBound];
}

void BackTrack(int nbItems, int knapsackBound, unsigned int** matrixDP,
               vector<bool>& solution, vector<int>& weights) {
  if (verboseMode) {
    cout << "print DP matrix : " << endl;
    for (int i = 0; i < nbItems; i++) {
      for (int j = 0; j <= knapsackBound; j++) cout << matrixDP[i][j] << " ";
      cout << endl;
    }
  }

  solution.clear();

  solution.resize(nbItems);

  int m = knapsackBound;
  for (int i = nbItems - 1; i >= 1; i--) {
    if (m < weights[i] || matrixDP[i][m] == matrixDP[i - 1][m])
      solution[i] = false;
    else {
      solution[i] = true;
      m -= weights[i];
    }
  }

  if (m < weights[0])
    solution[0] = false;
  else
    solution[0] = true;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: knapsack inputFile  [verbose] " << endl;
    cerr << "A second optional allows to disable the verbose mode for debugging"
         << endl;
    return 1;
  }

  vector<int> weights;
  vector<int> values;
  int knapsackBound = 0;
  int nbItems;
  int costSolution = 0;
  vector<bool> solution;

  unsigned int** matrixDP;

  double totaltime = 0;

  // if (argc < 3) nbMaxIt
  // ems = atoi(argv[3]);
  if (argc < 3) verboseMode = false;
  const char* instanceFile = argv[1];

  readInstance(instanceFile, weights, values, knapsackBound, nbItems);

  // InitializeMatrixDP(matrixDP,nbItems,knapsackBound, weights,values);
  matrixDP = new unsigned int*[nbItems];
  for (int i = 0; i < nbItems; i++) {
    matrixDP[i] = new unsigned int[knapsackBound + 1];
    for (int j = 0; j <= knapsackBound; j++) matrixDP[i][j] = 0;
  }

  for (int m = 0; m <= knapsackBound; m++)
    if (m < weights[0])
      matrixDP[0][m] = 0;
    else
      matrixDP[0][m] = values[0];
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  int rankID;
  MPI_Comm_rank(MPI_COMM_WORLD, &rankID);
  int nbprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nbprocs);

  auto start = std::chrono::steady_clock::now();
  DistributedDP(weights, values, knapsackBound, nbItems, costSolution, solution,
                matrixDP, rankID, nbprocs);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  double count = elapsed_seconds.count();
  MPI_Allreduce(&count, &totaltime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (rankID == 0) {
    BackTrack(nbItems, knapsackBound, matrixDP, solution, weights);
    cout << "solution optimale trouvee de cout " << costSolution
         << " en temps: " << elapsed_seconds.count() << "s" << endl
         << endl;
    if (verboseMode) printKnapsackSolution(solution);
  }

  MPI_Finalize();

  // destruction de la matrice de programmation dynamique
  for (int i = 0; i < nbItems; i++) delete[] matrixDP[i];
  delete[] matrixDP;

  return 0;
}