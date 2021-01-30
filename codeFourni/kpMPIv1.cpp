#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <mpi.h>
#include<math.h>


using namespace std;

bool verboseMode = true;

//lit une instance donnée par le nom du fichier  fileName, et crée le sac à dos correspondant dans les tableaux donnés par pointeur 
void readInstance(const string& fileName, vector<int> & weights, vector<int> &values,  int & knapsackBound,  int & nbItems) {

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
	for (int i = 0; i < nbItems; i++) cout << "item: " << i << " weight: " << weights[i] << " value: " << values[i] << endl;
*/

}


void printKnapsackSolution(vector<bool> & solution) {
	cout << "knapsack composition  : ";
	for (std::vector<bool>::iterator it = solution.begin() ; it != solution.end(); ++it)
		    std::cout << ' ' << *it;
	cout  << endl;
}

void DistributedDP(vector<int> & weights, vector<int> & values, int knapsackBound,  int  nbItems, int & costSolution, vector<bool> & solution,
int nbprocs,int ChargeUnit,unsigned int** matrixDP,int rankID){
  
  int *recv_buffer;
  int *send_buffer;
 for(int i = 1; i < nbItems;i++){ // each objet
    
     send_buffer = new int[ChargeUnit];
     
     recv_buffer = new int[nbprocs * ChargeUnit];

     MPI_Barrier(MPI_COMM_WORLD);
        for(int x=0;x<ChargeUnit;x++){
	  int m = rankID*ChargeUnit + (x+1) ;
	   //cout<<"("<<rankID<<","<<x<<","<<m<<")"<<",";
	  if (weights[i] <= m){
	   /*   if(m>knapsackBound){ //如果 m 超出了表的范围
                 send_buffer[x] = 0; 
                
		 }else{*/ //如果 m 在范围内
	    send_buffer[x] = max(values[i] + matrixDP[i-1][m - weights[i]],  matrixDP[i-1][m]);
		//x}
          }else{
	    send_buffer[x] =  matrixDP[i-1][m];
        }
     }
     MPI_Allgather(send_buffer,ChargeUnit,MPI_INT,recv_buffer,
                    ChargeUnit,MPI_INT,MPI_COMM_WORLD); 
   
        for(int j = 1;j<=knapsackBound ; j++){
            matrixDP[i][j] = recv_buffer[j-1];
        }
     
  }

   costSolution = matrixDP[nbItems - 1][knapsackBound];//最后一行的最后一个值 
	
}

void BackTrack(int nbItems,int knapsackBound,unsigned int** matrixDP,  vector<bool> & solution, vector<int> & weights){
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
      if (m < weights[i] || matrixDP[i][m] == matrixDP[i - 1][m]) solution[i] = false;//当总值相同时优先考虑放入商品数量少的解
      else {//说明在不超过承重量条件下放入了第i个物品
	solution[i] = true;
	m -= weights[i];
      }
    }

    if (m < weights[0]) solution[0] = false;
    else solution[0] = true;
}


int main(int argc,char** argv){
  if (argc < 2) {
    cerr << "Usage: knapsack inputFile  [verbose] " << endl;
    cerr << "A second optional allows to disable the verbose mode for debugging" << endl;
    return 1;
  }

  vector<int> weights;
  vector<int> values;
  int knapsackBound = 0;
  int nbItems;
  int costSolution = 0 ;
  vector<bool> solution;

  int ChargeUnit;
  unsigned int** matrixDP;

  double totaltime=0;
 

  //if (argc < 3) nbMaxIt
  // ems = atoi(argv[3]);
  if (argc < 3) verboseMode = false;
    const char* instanceFile = argv[1];

    readInstance(instanceFile, weights, values, knapsackBound, nbItems);

   

  //InitializeMatrixDP(matrixDP,nbItems,knapsackBound, weights,values);
  matrixDP = new unsigned int* [nbItems];
  for(int i = 0; i < nbItems; i++){
    matrixDP[i] = new unsigned int [knapsackBound+1];
    for(int j = 0; j <= knapsackBound; j++) matrixDP[i][j] =0;
  }

  
  for(int m = 0; m <= knapsackBound; m++) if (m <weights[0]) matrixDP[0][m] =0; else matrixDP[0][m] =values[0];
 
  //Initialize the MPI environment
  MPI_Init(NULL,NULL);  
 
  int rankID;
  MPI_Comm_rank(MPI_COMM_WORLD,&rankID);
  int nbprocs;
  MPI_Comm_size(MPI_COMM_WORLD,&nbprocs);

  ChargeUnit =ceil((double)knapsackBound / (double)nbprocs); //each process take charge cases length 
  
  //std::cout<<"chargeUnit: "<<ChargeUnit<<std::endl;
  
  auto start = std::chrono::steady_clock::now();

  DistributedDP(weights, values, knapsackBound,  nbItems,costSolution,solution,nbprocs,ChargeUnit,
  	   matrixDP,rankID);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double count = elapsed_seconds.count();
  MPI_Allreduce(&count,&totaltime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  if(rankID == 0){

    BackTrack(nbItems,knapsackBound,matrixDP,solution,weights);
    cout << "solution optimale trouvee de cout " << costSolution << " en temps: " << elapsed_seconds.count() << "s" << endl<< endl;
    printKnapsackSolution(solution);
    //cout << "print DP matrix : " << endl;
  }

  MPI_Finalize();

  //destruction de la matrice de programmation dynamique
  for (int i = 0; i < nbItems; i++) delete[] matrixDP[i];
  delete[] matrixDP;
 
 


  return 0;

  

}
