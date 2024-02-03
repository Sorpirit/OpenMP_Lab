#pragma once
#include <mpi.h>

#include "IAlgorithm.h"

class CollectiveQue : IAlgorithm
{
public:
	CollectiveQue(int rank, int numtasks, int rCount, int cCount);
	~CollectiveQue();
	void runMaster() override;
	void runWorker() override;
private:


	const int rank;
	int numtasks;
	int numworkers;

	const int rowsCount;
	const int colsCount;

	MPI_Status status;
	MPI_Request request;
	int rows;

	double* a;
	double* b;
	double* c;
};
