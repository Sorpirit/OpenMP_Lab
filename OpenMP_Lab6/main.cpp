#include <mpi.h>
#include <cstdio>
#include <cstdlib>

#include "Constants.h"
#include "IAlgorithm.h"
#include "BlockingQue.h"
#include "NonBlockingQue.h"
#include "CollectiveQue.h"

int main(int argc, char* argv[]) {
	int numtasks = 0;
	int	taskid = 0;
	int rc = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	if (numtasks < 2) {
		printf("Need at least two MPI tasks. Quitting...\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
		exit(1);
	}

	auto algorithm = CollectiveQue(taskid, numtasks, 100, 100);
	if(taskid == MASTER)
	{
		algorithm.runMaster();
	}
	else
	{
		algorithm.runWorker();
	}
	
	MPI_Finalize();
}