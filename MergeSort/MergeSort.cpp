// MergeSort.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <stdio.h>
#include <memory.h>
#include <stdlib.h>

#define ARRAY_LEN 10
#define INFINITE 9999

int Arr[] = {34,23,55,1,5,76,92,21,66,99};

void printArray();
void mergeSort(int nStart, int nEnd);
void merge(int nStart, int nCenter, int nEnd);

int main()
{
	printf("Unsorted Array:\t");
	printArray();
	mergeSort(0, ARRAY_LEN - 1);
	printf("Sorted Array:\t");
	printArray();
	return 0;
}

void printArray()
{
	for (int i : Arr)
	{
		printf("%d ", i);
	}
	printf("\n");
}

void mergeSort(int nStart, int nEnd)
{
	if (nStart < nEnd)
	{
		int nCenter = (nEnd + nStart) / 2;
		mergeSort(nStart, nCenter);
		mergeSort(nCenter+1 , nEnd);
		merge(nStart, nCenter, nEnd);
	}

}

int* GetArray(int nStart, int nEnd, int nLen)
{
	int* pArr = (int*)malloc(sizeof(int) * nLen);
	
	int j = 0;
	for (int i = nStart; i <= nEnd; i++)
	{
		pArr[j] = Arr[i];
		j++;
	}

	return pArr;
}
void merge(int nStart, int nCenter, int nEnd)
{
	int LLen = nCenter - nStart + 1;
	int RLen = nEnd - nCenter;

	int* L = GetArray(nStart, nCenter, LLen+1);
	int* R = GetArray(nCenter+1, nEnd, RLen+1);
	L[LLen] = INFINITE;
	R[RLen] = INFINITE;

	int i = 0;
	int j = 0;

	for (int k = nStart; k <= nEnd; k++)
	{
		if (L[i] <= R[j])
		{
			Arr[k] = L[i];
			i++;
		}
		else
		{
			Arr[k] = R[j];
			j++;
		}
	}
	free(L);
	free(R);
}

