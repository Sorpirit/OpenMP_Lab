#include "CollectiveQue.h"
#include "Constants.h"

#include <cstdio>

CollectiveQue::CollectiveQue(int rank, int numtasks, int rCount, int cCount)
	: rank(rank), numtasks(numtasks), numworkers(numtasks), rowsCount(rCount), colsCount(cCount)
{
}

CollectiveQue::~CollectiveQue()
{
	delete[] a;
	delete[] b;
	delete[] c;
}

void CollectiveQue::runMaster()
{
	int i, j, k;

	a = new double[rowsCount * colsCount];
	b = new double[rowsCount * colsCount];
	c = new double[rowsCount * colsCount];

	printf("running CollectiveQue\n");
	printf("mpi_mm has started with %d tasks and %d workers.\n", numtasks, numworkers);
	for (i = 0; i < rowsCount; i++)
		for (j = 0; j < colsCount; j++)
			a[i * rowsCount + j] = 10;
	for (i = 0; i < rowsCount; i++)
		for (j = 0; j < colsCount; j++)
			b[i * rowsCount + j] = 10;

	int* counts = new int[numtasks];
	int* displs = new int[numtasks];

	int averow = rowsCount / numtasks;
	int extra = rowsCount % numtasks;

	int offset = 0;
	for (int destination = 0; destination < numtasks; destination++) {
		rows = averow;
		if (destination == numtasks - 1)
			rows += extra;

		counts[destination] = rows * colsCount;
		displs[destination] = offset;

		offset += rows * colsCount;
	}

	double startTime = MPI_Wtime() * 1000.0;

 	double* receive_a = new double[counts[rank]];
 	double* sned_c = new double[counts[rank]];
	MPI_Scatterv(a, counts, displs, MPI_DOUBLE, receive_a, counts[rank], MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(b, rowsCount * colsCount, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

	rows = counts[rank] / colsCount;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < colsCount; j++)
		{
			sned_c[i * colsCount + j] = 0.0;
			for (k = 0; k < rowsCount; k++)
			{
				sned_c[i * colsCount + j] += receive_a[i * colsCount + k] * b[k * colsCount + j];
			}
		}
	}
	/*MPI_Gatherv(sned_c, counts[rank], MPI_DOUBLE, &c[displs[rank]],
		counts, displs, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);*/
	MPI_Allgatherv(sned_c, counts[rank], MPI_DOUBLE, c,
		counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

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

	delete[] counts;
	delete[] displs;
}

void CollectiveQue::runWorker()
{
	int i, j, k;

	a = new double[rowsCount * colsCount];
	b = new double[rowsCount * colsCount];
	c = new double[rowsCount * colsCount];

	int* counts = new int[numtasks];
	int* displs = new int[numtasks];

	int averow = rowsCount / numtasks;
	int extra = rowsCount % numtasks;

	int offset = 0;
	for (int destination = 0; destination < numtasks; destination++) {
		rows = averow;
		if (destination == numtasks - 1)
			rows += extra;

		counts[destination] = rows * colsCount;
		displs[destination] = offset;

		offset += rows * colsCount;
	}

	double* sned_c = new double[counts[rank]];

	MPI_Scatterv(a, counts, displs, MPI_DOUBLE, a, counts[rank], MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(b, rowsCount * colsCount, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	rows = counts[rank] / colsCount;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < colsCount; j++)
		{
			sned_c[i * colsCount + j] = 0.0;
			for (k = 0; k < rowsCount; k++)
			{
				sned_c[i * colsCount + j] += a[i * colsCount + k] * b[k * colsCount + j];
			}
		}
	}

	/*MPI_Gatherv(sned_c, counts[rank], MPI_DOUBLE, &c[displs[rank]],
		counts, displs, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);*/
	MPI_Allgatherv(sned_c, counts[rank], MPI_DOUBLE, c,
		counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
}
