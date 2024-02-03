#pragma once

#include <mpi.h>
#include "IAlgorithm.h"
#include "Constants.h"

class BlockingQue : public IAlgorithm
{
public:
	BlockingQue(int rank, int numtasks, int rCount, int cCount);
	~BlockingQue();

	void runMaster() override;
	void runWorker() override;
private:
	

	int rank;
	int numtasks;
	int numworkers;

	int rowsCount;
	int colsCount;

	MPI_Status status;
	int rows;

	double* a;
	double* b;
	double* c;
};
