#include "BlockingQue.h"

#include <cstdio>
#include <ctime>

BlockingQue::BlockingQue(int rank, int numtasks, int rCount, int cCount)
	: rank(rank), numtasks(numtasks), numworkers(numtasks - 1), rowsCount(rCount), colsCount(cCount)
{
}

BlockingQue::~BlockingQue()
{
	delete[] a;
	delete[] b;
	delete[] c;
}

void BlockingQue::runMaster()
{
	int i, j;

	a = new double[rowsCount * colsCount];
	b = new double[rowsCount * colsCount];
	c = new double[rowsCount * colsCount];

	printf("running BlockingQue\n");
	printf("mpi_mm has started with %d tasks and %d workers.\n", numtasks, numworkers);
	for (i = 0; i < rowsCount; i++)
		for (j = 0; j < colsCount; j++)
			a[i * rowsCount + j] = 10;
	for (i = 0; i < rowsCount; i++)
		for (j = 0; j < colsCount; j++)
			b[i * rowsCount + j] = 10;

	int averow = rowsCount / numworkers;
	int extra = rowsCount % numworkers;
	int offset = 0;
	int dest = 0;

	clock_t tStart = clock();
	double startTime = MPI_Wtime() * 1000.0;

	for (dest = 1; dest <= numworkers; dest++) {
		rows = averow;
		if (dest == numworkers)
			rows += extra;

		MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER,
			MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER,
			MPI_COMM_WORLD);
		MPI_Send(a + offset, rows * colsCount, MPI_DOUBLE, dest,
			FROM_MASTER, MPI_COMM_WORLD);
		MPI_Send(b, rowsCount * colsCount, MPI_DOUBLE, dest, FROM_MASTER,
			MPI_COMM_WORLD);
		offset += rows * colsCount;
	}

	/* Receive results from worker tasks */
	int source = 0;
	for (source = 1; source <= numworkers; source++) {
		MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER,
			MPI_COMM_WORLD, &status);
		MPI_Recv(&rows, 1, MPI_INT, source, FROM_WORKER,
			MPI_COMM_WORLD, &status);
		MPI_Recv(c + offset, rows * colsCount, MPI_DOUBLE,
			source, FROM_WORKER,
			MPI_COMM_WORLD, &status);
		printf("Received results from task %d %6.2f \n", source, c[offset]);
	}

	double endTime = MPI_Wtime() * 1000.0;
	clock_t tEnd = clock();
	double elapsedMs = (endTime - startTime);
	double elapsedMsClock = ((double)(tEnd - tStart) / CLOCKS_PER_SEC) * 1000;

	/* Print results */
	printf("****\n");
	printf("Result Matrix:\n");
	/*for (i = 0; i < rowsCount; i++) {
		printf("\n");
		for (j = 0; j < colsCount; j++)
			printf("%6.2f ", c[i * colsCount + j]);
	}*/
	printf("\n********\n");
	printf("Done. Elapsed: %6.2fms or %6.2f\n", elapsedMs, elapsedMsClock);
	
}

void BlockingQue::runWorker()
{
	int offset = 0;
	int i, j, k;

	b = new double[rowsCount * colsCount];

	MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER,
		MPI_COMM_WORLD, &status);
	MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER,
		MPI_COMM_WORLD, &status);

	a = new double[rows * colsCount];
	c = new double[rows * colsCount];

	MPI_Recv(a, rows * colsCount, MPI_DOUBLE, MASTER, FROM_MASTER,
		MPI_COMM_WORLD, &status);
	MPI_Recv(b, rowsCount * colsCount, MPI_DOUBLE, MASTER, FROM_MASTER,
		MPI_COMM_WORLD, &status);
	for (i = 0; i < rows; i++) 
	{
		for (j = 0; j < colsCount; j++)
		{
			c[i * colsCount + j] = 0.0;
			for (k = 0; k < rowsCount; k++)
			{
				c[i * colsCount + j] += a[i * colsCount + k] * b[k * colsCount + j];
			}
		}
	}
	MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER,
		MPI_COMM_WORLD);
	MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER,
		MPI_COMM_WORLD);
	MPI_Send(c, rows * colsCount, MPI_DOUBLE, MASTER,
		FROM_WORKER, MPI_COMM_WORLD);
}
