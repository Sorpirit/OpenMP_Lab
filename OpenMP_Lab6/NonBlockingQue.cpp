#include "NonBlockingQue.h"

#include <cstdio>

NonBlockingQue::NonBlockingQue(int rank, int numtasks, int rCount, int cCount)
	: rank(rank), numtasks(numtasks), numworkers(numtasks - 1), rowsCount(rCount), colsCount(cCount)
{
}

NonBlockingQue::~NonBlockingQue()
{
	delete[] a;
	delete[] b;
	delete[] c;
}

void NonBlockingQue::runMaster()
{
	int i, j;

	a = new double[rowsCount * colsCount];
	b = new double[rowsCount * colsCount];
	c = new double[rowsCount * colsCount];

	printf("running NonBlockingQue\n");
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

	auto requests = new MPI_Request[4 * numworkers];
	int reqs_offest = 0;
	auto recv_status = new MPI_Status[4 * numworkers];

	double startTime = MPI_Wtime() * 1000.0;

	for (dest = numworkers; dest >= 1; dest--) {
		rows = averow;
		if (dest == numworkers)
			rows += extra;

		reqs_offest = (dest - 1) * 4;
		MPI_Isend(&offset, 1, MPI_INT, dest, FROM_MASTER,
			MPI_COMM_WORLD, &requests[reqs_offest]);
		MPI_Isend(&rows, 1, MPI_INT, dest, FROM_MASTER,
			MPI_COMM_WORLD, &requests[reqs_offest + 1]);
		MPI_Isend(a + offset, rows * colsCount, MPI_DOUBLE, dest,
			FROM_MASTER, MPI_COMM_WORLD, &requests[reqs_offest + 2]);
		MPI_Isend(b, rowsCount * colsCount, MPI_DOUBLE, dest, FROM_MASTER,
			MPI_COMM_WORLD, &requests[reqs_offest + 3]);
		offset += rows * colsCount;
	}

	//MPI_Waitall(4 * numworkers, requests, recv_status);
	
	int source = 0;
	for (source = 1; source <= numworkers; source++) {
		printf("maseter getting...%d\n", source);

		MPI_Irecv(&offset, 1, MPI_INT, source, FROM_WORKER,
			MPI_COMM_WORLD, &requests[0]);
		MPI_Irecv(&rows, 1, MPI_INT, source, FROM_WORKER,
			MPI_COMM_WORLD, &requests[1]);

		MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
		MPI_Wait(&requests[1], MPI_STATUS_IGNORE);

		MPI_Irecv(c + offset, rows * colsCount, MPI_DOUBLE,
			source, FROM_WORKER,
			MPI_COMM_WORLD, &requests[2]);

		MPI_Wait(&requests[2], MPI_STATUS_IGNORE);

		printf("Received results from task %d %6.2f \n", source, c[offset]);
	}

	double endTime = MPI_Wtime() * 1000.0;
	double elapsedMs = (endTime - startTime);

	/* Print results */
	printf("****\n");
	printf("Result Matrix:\n");
	for (i = 0; i < rowsCount; i++) {
		printf("\n");
		for (j = 0; j < colsCount; j++)
			printf("%6.2f ", c[i * colsCount + j]);
	}
	printf("\n********\n");
	printf("Done. Elapsed: %6.2fms\n", elapsedMs);

	delete[] requests;
	delete[] recv_status;
}

void NonBlockingQue::runWorker()
{
	int offset = 0;
	int i, j, k;
	
	MPI_Request requests[3];
	MPI_Status recv_status[3];

	b = new double[rowsCount * colsCount];

	MPI_Irecv(&offset, 1, MPI_INT, MASTER, FROM_MASTER,
		MPI_COMM_WORLD, &requests[0]);
	MPI_Irecv(&rows, 1, MPI_INT, MASTER, FROM_MASTER,
		MPI_COMM_WORLD, &requests[1]);

	MPI_Waitall(2, requests, recv_status);

	MPI_Isend(&offset, 1, MPI_INT, MASTER, FROM_WORKER,
		MPI_COMM_WORLD, &requests[0]);
	MPI_Isend(&rows, 1, MPI_INT, MASTER, FROM_WORKER,
		MPI_COMM_WORLD, &requests[1]);

	a = new double[rows * colsCount];
	c = new double[rows * colsCount];

	MPI_Irecv(a, rows * colsCount, MPI_DOUBLE, MASTER, FROM_MASTER,
		MPI_COMM_WORLD, &requests[0]);
	MPI_Irecv(b, rowsCount * colsCount, MPI_DOUBLE, MASTER, FROM_MASTER,
		MPI_COMM_WORLD, &requests[1]);

	MPI_Waitall(2, requests, recv_status);

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

	
	MPI_Isend(c, rows * colsCount, MPI_DOUBLE, MASTER,
		FROM_WORKER, MPI_COMM_WORLD, &requests[2]);

	//MPI_Waitall(3, requests, recv_status);
}
