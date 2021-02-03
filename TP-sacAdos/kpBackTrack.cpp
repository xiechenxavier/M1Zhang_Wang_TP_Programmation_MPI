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

void printKnapsackSolution(vector<unsigned int>& solution) {
  cout << "knapsack composition  : ";
  for (std::vector<unsigned int>::iterator it = solution.begin(); it != solution.end();
       ++it)
    std::cout << ' ' << *it;
  cout << endl;
}

pair<int, int> BackTrack(int nbItems, int part_knapsackBound, int** local_matrix_full,
               vector<unsigned int>& solution, vector<int>& weights, int rankID, pair<int, int>& start_index) {
  solution.resize(nbItems);
  int m = part_knapsackBound * (rankID + 1) - 1;
  int j = start_index.second;
  int i;
  
  for (i = start_index.first; i >= 1; i--) {
    if (m < weights[i] || local_matrix_full[i][j + rankID * part_knapsackBound] == local_matrix_full[i - 1][j+ rankID * part_knapsackBound]) {
      solution[i] = 0;
    }
    else {
      solution[i] = 1;
      m -= weights[i];
      j -= weights[i];
      if (j < 0) return make_pair(i, part_knapsackBound + j - 1);
    }
  }
  if (m < weights[0])
    solution[0] = 0;
  else
    solution[0] = 1;
  return make_pair(i - 1, part_knapsackBound + j);
}

void DistributedDP(vector<int>& weights, vector<int>& values, int knapsackBound,
                   int nbItems, int& costSolution, vector<unsigned int>& global_solution,
                   unsigned int** matrixDP, int rankID, int nbprocs) {
  MPI_Barrier(MPI_COMM_WORLD);
  int index_i;
  int index_j;
  MPI_Status status;
  vector<unsigned int> local_solution;
  int ceil_charge_unit = ceil((double)knapsackBound / (double)nbprocs);
  int local_matrix[ceil_charge_unit];
  int** local_matrix_full;
  int vec_buffer[nbprocs * ceil_charge_unit];
  local_matrix_full = new int*[nbItems];
  for (int k = 0; k < nbItems; k++) local_matrix_full[k] = new int[ceil_charge_unit];
  for (int k = 0; k < ceil_charge_unit; k++) {
    local_matrix[k] = matrixDP[0][k + rankID * ceil_charge_unit + 1];
    local_matrix_full[0][k] = matrixDP[0][k + rankID * ceil_charge_unit + 1];
  }
  for (int k = 0; k < knapsackBound; k++) {
      vec_buffer[k] = matrixDP[0][k+1];
  }
  for (int i = 1; i < nbItems; i++) {
    for (int j = 0; j < ceil_charge_unit; j++) {
      int m = rankID * ceil_charge_unit + j + 1;
      if (weights[i] <= m) {
        local_matrix[j] = max(
            values[i] + vec_buffer[max((m - weights[i] - 1), 0)], vec_buffer[m - 1]);
      } else {
        local_matrix[j] = vec_buffer[m - 1];
      }
    }
    MPI_Allgather(local_matrix, ceil_charge_unit, MPI_INT, vec_buffer, ceil_charge_unit,
                  MPI_INT, MPI_COMM_WORLD);
    for (int k = 1; k < knapsackBound + 1; k++) {
      matrixDP[i][k] = vec_buffer[k-1];
    }
    for (int k = 0; k < ceil_charge_unit; k++) {
      local_matrix_full[i][k] = matrixDP[i][k + rankID * ceil_charge_unit + 1];
    }
  }
  if (rankID == nbprocs - 1) {
      pair<int, int> start_index (nbItems - 1, ceil_charge_unit - 1);
      pair<int, int> index(BackTrack(nbItems, ceil_charge_unit, local_matrix_full, local_solution, weights, rankID, start_index));
      MPI_Send(&(index.first), 1, MPI_INT, rankID - 1, 0, MPI_COMM_WORLD);
      MPI_Send(&(index.second), 1, MPI_INT, rankID - 1, 1, MPI_COMM_WORLD);
  } else if (rankID == 0) {
      MPI_Recv(&index_i, 1, MPI_INT, rankID + 1, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&index_j, 1, MPI_INT, rankID + 1, 1, MPI_COMM_WORLD, &status);
      pair<int, int> start_index(index_i, index_j);
      BackTrack(nbItems, ceil_charge_unit, local_matrix_full, local_solution, weights, rankID, start_index);
  } else {
      MPI_Recv(&index_i, 1, MPI_INT, rankID + 1, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&index_j, 1, MPI_INT, rankID + 1, 1, MPI_COMM_WORLD, &status);
      pair<int, int> start_index(index_i, index_j);
      pair<int, int> index (BackTrack(nbItems, ceil_charge_unit, local_matrix_full, local_solution, weights, rankID, start_index));
      MPI_Send(&(index.first), 1, MPI_INT, rankID - 1, 0, MPI_COMM_WORLD);
      MPI_Send(&(index.second), 1, MPI_INT, rankID - 1, 1, MPI_COMM_WORLD);
  }
  global_solution.resize(nbItems);
  MPI_Allreduce(local_solution.data(), global_solution.data(), nbItems, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
  for (int k = 0; k < nbItems; k++) delete[] local_matrix_full[k];
  delete[] local_matrix_full;
  costSolution = matrixDP[nbItems - 1][knapsackBound];
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
  vector<unsigned int> global_solution;

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
  DistributedDP(weights, values, knapsackBound, nbItems, costSolution, global_solution,
                matrixDP, rankID, nbprocs);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  double count = elapsed_seconds.count();
  MPI_Allreduce(&count, &totaltime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (rankID == 0) {
    if (verboseMode) {
      cout << "print DP matrix : " << endl;
      for (int i = 0; i < nbItems; i++) {
        for (int j = 0; j <= knapsackBound; j++) cout << matrixDP[i][j] << " ";
        cout << endl;
      }
    }
    cout << "solution optimale trouvee de cout " << costSolution
         << " en temps: " << elapsed_seconds.count() << "s" << endl
         << endl;
    if (verboseMode) printKnapsackSolution(global_solution);
  }

  MPI_Finalize();

  // destruction de la matrice de programmation dynamique
  for (int i = 0; i < nbItems; i++) delete[] matrixDP[i];
  delete[] matrixDP;

  return 0;
}