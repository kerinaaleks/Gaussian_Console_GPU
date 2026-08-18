#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

using namespace std;

constexpr int THREADS_PER_BLOCK = 256;

// ============================================================================
// ЯДРА РАСПАКОВКИ
// ============================================================================

__global__ void UnpackBinKernel(const uint8_t* __restrict__ fileData, uint8_t* __restrict__ d_fastRows,
	size_t codeLength, size_t frameLength, size_t wordsCount)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= wordsCount * codeLength) return;

	size_t i = idx / codeLength;
	size_t b = idx % codeLength;
	size_t bitStart = i * codeLength + b;

	if ((fileData[bitStart / 8] >> (bitStart % 8)) & 1) {
		atomicOr(&d_fastRows[i * frameLength + (b / 8)], (uint8_t)(1u << (b % 8)));
	}
}

__global__ void UnpackBisKernel(
	const uint8_t* __restrict__ fileData,
	uint8_t* __restrict__ d_fastRows,
	size_t codeLength,
	size_t frameLength,
	size_t wordsCount)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t totalBits = wordsCount * codeLength;
	if (idx >= totalBits) return;

	size_t i = idx / codeLength;
	size_t b = idx % codeLength;

	if (fileData[i * codeLength + b] != 0) {
		atomicOr(&d_fastRows[i * frameLength + (b / 8)], (uint8_t)(1u << (b % 8)));
	}
}

// ============================================================================
// ЯДРА ГАУССА
// ============================================================================

__global__ void FindLeadingOneKernel(const uint8_t* __restrict__ rows, int* __restrict__ pivots,
	int frameLength, int wordsCount)
{
	int laneId = threadIdx.x & 31;
	int warpId = threadIdx.x >> 5;
	int warpsPerBlock = blockDim.x >> 5;
	int rowIndex = blockIdx.x * warpsPerBlock + warpId;
	if (rowIndex >= wordsCount) return;

	const uint8_t* row = rows + rowIndex * frameLength;
	int localPivot = INT_MAX;

	for (int b = laneId; b < frameLength; b += 32) {
		uint8_t val = row[b];
		if (val != 0) {
			for (int bit = 0; bit < 8; bit++) {
				if (val & (1 << bit)) {
					localPivot = min(localPivot, b * 8 + bit);
					break;
				}
			}
		}
	}

	for (int offset = 16; offset > 0; offset >>= 1) {
		int other = __shfl_down_sync(0xFFFFFFFFu, localPivot, offset);
		localPivot = min(localPivot, other);
	}

	if (laneId == 0)
		pivots[rowIndex] = (localPivot == INT_MAX) ? -1 : localPivot;
}

__global__ void FindPivotInSingleRow(
	const uint8_t* row,
	int frameLength,
	int* pivotOut)
{
	int tid = threadIdx.x;
	int laneId = tid & 31;
	int warpId = tid >> 5;
	__shared__ int warpMin[THREADS_PER_BLOCK / 32];

	int localPivot = INT_MAX;

	for (int b = tid; b < frameLength; b += blockDim.x) {
		uint8_t v = row[b];
		if (v != 0) {
			for (int bit = 0; bit < 8; ++bit) {
				if (v & (1 << bit)) {
					localPivot = min(localPivot, b * 8 + bit);
					break;
				}
			}
		}
	}

	for (int offset = 16; offset > 0; offset >>= 1)
		localPivot = min(localPivot, __shfl_down_sync(0xFFFFFFFFu, localPivot, offset));

	if (laneId == 0)
		warpMin[warpId] = localPivot;

	__syncthreads();

	if (tid == 0) {
		int finalPivot = INT_MAX;
		for (int i = 0; i < (blockDim.x >> 5); ++i)
			finalPivot = min(finalPivot, warpMin[i]);
		*pivotOut = (finalPivot == INT_MAX) ? -1 : finalPivot;
	}
}

__global__ void XorRowsKernel(
	uint8_t* d_rows,
	uint8_t* d_matrix,
	int frameLength,
	int srcRowIdx,
	int targetRow,
	int codeLength)
{
	uint8_t* src = d_rows + srcRowIdx * frameLength;
	uint8_t* tgt = d_matrix + targetRow * frameLength;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int numU64 = frameLength / 8;
	uint64_t* src64 = reinterpret_cast<uint64_t*>(src);
	uint64_t* tgt64 = reinterpret_cast<uint64_t*>(tgt);

	for (int i = tid; i < numU64; i += stride) {
		src64[i] ^= tgt64[i];
	}

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		int tailStart = numU64 * 8;
		for (int i = tailStart; i < frameLength; ++i) {
			src[i] ^= tgt[i];
		}
		int rem = codeLength % 8;
		if (rem != 0)
			src[frameLength - 1] &= (uint8_t)((1u << rem) - 1u);
	}
}

__device__ __forceinline__ bool GetBitDevice(const uint8_t* row, int bitIndex)
{
	return (row[bitIndex / 8] >> (bitIndex % 8)) & 1;
}

inline bool GetBitHost(const uint8_t* row, size_t bitIndex)
{
	return (row[bitIndex / 8] >> (bitIndex % 8)) & 1;
}

__global__ void ReverseGearKernel(
	uint8_t* d_matrix,
	int matrixRows,
	int frameLength,
	int pivotRow,
	int* d_pivotsMatrix,
	int codeLength)
{
	int j = blockIdx.x;
	if (j >= matrixRows || j == pivotRow) return;

	uint8_t* rowJ = d_matrix + j * frameLength;
	const uint8_t* rowI = d_matrix + pivotRow * frameLength;

	int pivotColumn = d_pivotsMatrix[pivotRow];
	if (pivotColumn < 0) return;
	if (!GetBitDevice(rowJ, pivotColumn)) return;

	for (int idx = threadIdx.x; idx < frameLength; idx += blockDim.x) {
		rowJ[idx] ^= rowI[idx];
		if (idx == frameLength - 1) {
			int rem = codeLength % 8;
			if (rem != 0)
				rowJ[idx] &= (uint8_t)((1u << rem) - 1u);
		}
	}
}

__global__ void FindPivotKernel(const int* pivotsMatrix, int matrixRows, int pivot, int* result)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= matrixRows) return;
	if (pivotsMatrix[tid] == pivot)
		atomicMin(result, tid);
}

// ============================================================================
// CPU-помощники
// ============================================================================

void ReverseGearGPU(
	uint8_t* d_matrix,
	size_t matrixRows,
	size_t frameLength,
	int* d_pivotsMatrix,
	int codeLength)
{
	for (int i = (int)matrixRows - 1; i >= 0; --i) {
		if (i > 0)
			ReverseGearKernel << <i, THREADS_PER_BLOCK >> > (
				d_matrix, (int)matrixRows, (int)frameLength,
				i, d_pivotsMatrix, codeLength);
	}
	cudaDeviceSynchronize();
}

// ============================================================================
// Упрощенная логика Гаусса без кэша
// ============================================================================

void ProcessAllFrameWithoutCache(
	uint8_t** matrix,
	uint8_t* d_rows, uint8_t* d_matrix,
	size_t wordsCount, size_t codeLength, size_t frameLength,
	size_t& matrixRows, size_t infoLength,
	int* cwIndex, int* pivotsMatrix,
	int* d_pivotsMatrix, int* d_foundRow, int* d_pivotOut)
{
	matrixRows = 0;

	for (int rowIndex = 0; rowIndex < (int)infoLength; ++rowIndex) {
		bool found = false;

		while (!found) {
			// 1. Ищем строку в d_rows с точным совпадением ведущего элемента (pivot == rowIndex)
			int selectedRowIndex = -1;
			for (size_t i = 0; i < wordsCount; ++i) {
				if (cwIndex[i] == rowIndex) {
					selectedRowIndex = (int)i;
					found = true;
					break;
				}
			}

			if (found) {
				// Копируем найденную строку в итоговую матрицу на позицию matrixRows
				cudaMemcpy(d_matrix + matrixRows * frameLength,
					d_rows + selectedRowIndex * frameLength,
					frameLength, cudaMemcpyDeviceToDevice);

				pivotsMatrix[matrixRows] = rowIndex;
				cudaMemcpy(d_pivotsMatrix + matrixRows, &rowIndex, sizeof(int), cudaMemcpyHostToDevice);

				matrixRows++;
				cwIndex[selectedRowIndex] = -1; // Помечаем как отработанную
				break;
			}

			// 2. Если точного совпадения нет, ищем строку с наименьшим пивотом > rowIndex для редукции
			int minLargerPivot = INT_MAX;
			int candidateRow = -1;

			for (size_t i = 0; i < wordsCount; ++i) {
				if (cwIndex[i] > rowIndex && cwIndex[i] < minLargerPivot) {
					minLargerPivot = cwIndex[i];
					candidateRow = (int)i;
				}
			}

			if (candidateRow != -1) {
				// Ищем в уже построенной части d_matrix строку, у которой пивот равен minLargerPivot
				int init = INT_MAX;
				cudaMemcpy(d_foundRow, &init, sizeof(int), cudaMemcpyHostToDevice);

				int blocks = (int)((matrixRows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
				if (blocks < 1) blocks = 1;

				FindPivotKernel << <blocks, THREADS_PER_BLOCK >> > (
					d_pivotsMatrix, (int)matrixRows, minLargerPivot, d_foundRow);

				int target_row;
				cudaMemcpy(&target_row, d_foundRow, sizeof(int), cudaMemcpyDeviceToHost);

				if (target_row != INT_MAX) {
					// Делаем XOR текущей строки с уже готовой базисной строкой
					int n64 = frameLength / 8;
					int blocksXor = (n64 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					if (blocksXor < 1) blocksXor = 1;

					XorRowsKernel << <blocksXor, THREADS_PER_BLOCK >> > (
						d_rows, d_matrix, (int)frameLength, candidateRow, target_row, (int)codeLength);

					// Пересчитываем пивот для измененной строки
					FindPivotInSingleRow << <1, THREADS_PER_BLOCK >> > (
						d_rows + candidateRow * frameLength, (int)frameLength, d_pivotOut);

					cudaMemcpy(&cwIndex[candidateRow], d_pivotOut, sizeof(int), cudaMemcpyDeviceToHost);
					continue; // Повторяем проверку для текущего rowIndex
				}
			}

			// 3. Если ничего подходящего не нашли — выходим из цикла поиска для этого rowIndex
			break;
		}
	}

	// Обратный ход выполняется только для реально найденных строк (matrixRows)
	ReverseGearGPU(d_matrix, matrixRows, frameLength, d_pivotsMatrix, (int)codeLength);

	for (size_t i = 0; i < matrixRows; ++i)
		cudaMemcpy(matrix[i], d_matrix + i * frameLength, frameLength, cudaMemcpyDeviceToHost);
}

// ============================================================================
// Запись результата
// ============================================================================

void WriteResultToFile(
	const string& fileName,
	uint8_t** matrix,
	size_t matrixRows,
	size_t codeLength,
	bool isBis)
{
	ofstream out(fileName, ios::binary);
	if (!out.is_open()) return;

	if (isBis) {
		size_t totalBytes = matrixRows * codeLength;
		uint8_t* buf = new uint8_t[totalBytes]();
		size_t pos = 0;
		for (size_t row = 0; row < matrixRows; ++row) {
			for (size_t b = 0; b < codeLength; ++b) {
				if (GetBitHost(matrix[row], b))
					buf[pos] = 0xFF;
				++pos;
			}
		}
		out.write(reinterpret_cast<char*>(buf), totalBytes);
		delete[] buf;
	}
	else {
		size_t totalBits = matrixRows * codeLength;
		size_t totalBytes = (totalBits + 7) / 8;
		uint8_t* buf = new uint8_t[totalBytes]();
		size_t outBitPos = 0;
		for (size_t row = 0; row < matrixRows; ++row) {
			for (size_t b = 0; b < codeLength; ++b) {
				if (GetBitHost(matrix[row], b)) {
					buf[outBitPos / 8] |= (1u << (outBitPos % 8));
				}
				++outBitPos;
			}
		}
		out.write(reinterpret_cast<char*>(buf), totalBytes);
		delete[] buf;
	}
	out.close();
}

// ============================================================================
// MAIN
// ============================================================================

int main()
{
	setlocale(LC_ALL, "");
	auto start = chrono::high_resolution_clock::now();

	const string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\6.21.bin)";
	const string TempDir = R"(D:\Rubin\sessions\tmp_1783328386069\)";

	const size_t codeLength = 18004;
	const size_t infoLength = 16384;
	const size_t rawFrameLength = (codeLength + 7) / 8;
	const size_t frameLength = ((rawFrameLength + 7) / 8) * 8;
	const char dataType = 0;

	ifstream file(InputFileName, ios::binary);
	if (!file.is_open()) return 1;

	file.seekg(0, ios::end);
	size_t fileSizeBytes = file.tellg();
	file.seekg(0, ios::beg);

	size_t wordsCount = dataType ? (fileSizeBytes / codeLength) : (fileSizeBytes * 8 / codeLength);
	size_t totalBytes = wordsCount * frameLength;

	uint8_t** matrix = new uint8_t*[infoLength];
	for (size_t i = 0; i < infoLength; ++i)
		matrix[i] = new uint8_t[frameLength]();

	uint8_t* fileData = new uint8_t[fileSizeBytes];
	file.read(reinterpret_cast<char*>(fileData), fileSizeBytes);
	file.close();

	uint8_t* d_fileData = nullptr;
	cudaMalloc(&d_fileData, fileSizeBytes);
	cudaMemcpy(d_fileData, fileData, fileSizeBytes, cudaMemcpyHostToDevice);
	delete[] fileData;

	int* cwIndex = new int[wordsCount];
	for (size_t i = 0; i < wordsCount; ++i) cwIndex[i] = -1;

	int* pivotsMatrix = new int[infoLength];
	for (size_t i = 0; i < infoLength; ++i) pivotsMatrix[i] = -1;

	uint8_t *d_rows = nullptr, *d_matrix = nullptr;
	int *d_pivots = nullptr, *d_pivotsMatrix = nullptr;
	int *d_foundRow = nullptr, *d_pivotOut = nullptr;

	cudaMalloc(&d_rows, totalBytes);
	cudaMemset(d_rows, 0, totalBytes);

	cudaMalloc(&d_matrix, infoLength * frameLength);
	cudaMemset(d_matrix, 0, infoLength * frameLength);

	size_t totalThreads = wordsCount * codeLength;
	int blocksUnpack = (int)((totalThreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

	if (dataType) {
		UnpackBisKernel << <blocksUnpack, THREADS_PER_BLOCK >> > (d_fileData, d_rows, codeLength, frameLength, wordsCount);
	}
	else {
		UnpackBinKernel << <blocksUnpack, THREADS_PER_BLOCK >> > (d_fileData, d_rows, codeLength, frameLength, wordsCount);
	}
	cudaDeviceSynchronize();
	cudaFree(d_fileData);

	{
		cudaMalloc(&d_pivots, wordsCount * sizeof(int));
		int warpsPerBlock = THREADS_PER_BLOCK / 32;
		int blocks = (int)((wordsCount + warpsPerBlock - 1) / warpsPerBlock);
		FindLeadingOneKernel << <blocks, THREADS_PER_BLOCK >> > (
			d_rows, d_pivots, (int)frameLength, (int)wordsCount);
		cudaDeviceSynchronize();
		cudaMemcpy(cwIndex, d_pivots, wordsCount * sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(d_pivots);
	}

	cudaMalloc(&d_pivotsMatrix, infoLength * sizeof(int));
	cudaMemcpy(d_pivotsMatrix, pivotsMatrix, infoLength * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_foundRow, sizeof(int));
	cudaMalloc(&d_pivotOut, sizeof(int));

	size_t matrixRows = 0;

	ProcessAllFrameWithoutCache(
		matrix, d_rows, d_matrix,
		wordsCount, codeLength, frameLength,
		matrixRows, infoLength,
		cwIndex, pivotsMatrix,
		d_pivotsMatrix, d_foundRow, d_pivotOut);

	string outName = TempDir + (dataType ? "result.bis" : "result.bin");
	WriteResultToFile(outName, matrix, matrixRows, codeLength, dataType != 0);

	for (size_t i = 0; i < infoLength; ++i) delete[] matrix[i];
	delete[] matrix;
	delete[] cwIndex;
	delete[] pivotsMatrix;

	cudaFree(d_rows);
	cudaFree(d_matrix);
	cudaFree(d_pivotsMatrix);
	cudaFree(d_foundRow);
	cudaFree(d_pivotOut);

	auto ms = chrono::duration_cast<chrono::milliseconds>(
		chrono::high_resolution_clock::now() - start).count();
	cout << "Время выполнения программы в мс: " << ms << endl;

	return 0;
}