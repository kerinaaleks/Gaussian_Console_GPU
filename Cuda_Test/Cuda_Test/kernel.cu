#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <climits>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

using namespace std;

constexpr int THREADS_PER_BLOCK = 256;

__global__ void FindLeadingOneKernel(
	const uint8_t* __restrict__ rows,
	int* __restrict__ pivots,
	int frameLength,
	int wordsCount)
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

void ReadCodeWordsBin(const uint8_t* fileData, size_t cwIndex,
	size_t codeLength, uint8_t* dst)
{
	size_t bitStart = cwIndex * codeLength;
	size_t byteStart = bitStart / 8;
	size_t bitOffset = bitStart % 8;
	size_t frameLength = (codeLength + 7) / 8;

	memset(dst, 0, frameLength);

	for (size_t i = 0; i < codeLength; ++i) {
		size_t srcByte = byteStart + (bitOffset + i) / 8;
		size_t srcBit = (bitOffset + i) % 8;
		if ((fileData[srcByte] >> srcBit) & 1)
			dst[i / 8] |= (1u << (i % 8));
	}

	size_t rem = codeLength % 8;
	if (rem != 0)
		dst[frameLength - 1] &= (uint8_t)((1u << rem) - 1u);
}

void ReadCodeWordsBis(const uint8_t* fileData, size_t cwIndex,
	size_t codeLength, uint8_t* dst)
{
	size_t frameLength = (codeLength + 7) / 8;
	const uint8_t* src = fileData + cwIndex * codeLength;

	memset(dst, 0, frameLength);

	for (size_t i = 0; i < codeLength; ++i) {
		if (src[i] != 0)
			dst[i / 8] |= (1u << (i % 8));
	}

	size_t rem = codeLength % 8;
	if (rem != 0)
		dst[frameLength - 1] &= (uint8_t)((1u << rem) - 1u);
}

__device__ int DevFindExactType(const int* pivots, const int* rowType,
	int n, int value, int wantType)
{
	__shared__ int best;
	if (threadIdx.x == 0) best = INT_MAX;
	__syncthreads();

	for (int i = threadIdx.x; i < n; i += blockDim.x) {
		if (rowType[i] == wantType && pivots[i] == value)
			atomicMin(&best, i);
	}
	__syncthreads();
	return best;
}

__device__ int DevFindLessType(const int* pivots, const int* rowType,
	int n, int rowIndex, int wantType)
{
	__shared__ int best;
	if (threadIdx.x == 0) best = INT_MAX;
	__syncthreads();

	for (int i = threadIdx.x; i < n; i += blockDim.x) {
		int p = pivots[i];
		if (rowType[i] == wantType && p != -1 && p < rowIndex)
			atomicMin(&best, i);
	}
	__syncthreads();
	return best;
}

__device__ void DevXor(uint8_t* src, const uint8_t* tgt, int frameLength, int codeLength)
{
	for (int i = threadIdx.x; i < frameLength; i += blockDim.x)
		src[i] ^= tgt[i];
	__syncthreads();

	if (threadIdx.x == 0) {
		int rem = codeLength % 8;
		if (rem != 0)
			src[frameLength - 1] &= (uint8_t)((1u << rem) - 1u);
	}
	__syncthreads();
}

__device__ int DevLeadingOne(const uint8_t* row, int codeLength)
{
	__shared__ int warpMin[THREADS_PER_BLOCK / 32];
	int tid = threadIdx.x;
	int local = INT_MAX;
	int numBytes = (codeLength + 7) / 8;

	for (int b = tid; b < numBytes; b += blockDim.x) {
		uint8_t v = row[b];
		if (b == numBytes - 1) {
			int rem = codeLength % 8;
			if (rem != 0)
				v &= (uint8_t)((1u << rem) - 1u);
		}
		if (v != 0) {
			for (int bit = 0; bit < 8; ++bit) {
				int bitIndex = b * 8 + bit;
				if (bitIndex >= codeLength) break;
				if (v & (1 << bit)) {
					local = min(local, bitIndex);
					break;
				}
			}
		}
	}

	for (int off = 16; off > 0; off >>= 1)
		local = min(local, __shfl_down_sync(0xFFFFFFFFu, local, off));

	if ((tid & 31) == 0)
		warpMin[tid >> 5] = local;
	__syncthreads();

	if (tid == 0) {
		int best = INT_MAX;
		for (int i = 0; i < (blockDim.x >> 5); ++i)
			best = min(best, warpMin[i]);
		warpMin[0] = best;
	}
	__syncthreads();

	return (warpMin[0] == INT_MAX) ? -1 : warpMin[0];
}

__global__ void ProcessOneRowKernel(
	int rowIndex,
	uint8_t* d_fastRows,
	uint8_t* d_matrix,
	int* d_cwIndexFast,
	int* d_rowType,
	int* d_pivotsMatrix,
	int* d_matrixRows,
	int wordsCount,
	int frameLength,
	int codeLength)
{
	for (;;) {
		// 1) pivot == rowIndex (сначала RAM=0, потом Cache=1)
		int sel = DevFindExactType(d_cwIndexFast, d_rowType, wordsCount, rowIndex, 0);
		if (sel == INT_MAX)
			sel = DevFindExactType(d_cwIndexFast, d_rowType, wordsCount, rowIndex, 1);

		if (sel != INT_MAX) {
			int mrows = d_matrixRows[0];
			uint8_t* dst = d_matrix + mrows * frameLength;
			const uint8_t* src = d_fastRows + sel * frameLength;

			for (int i = threadIdx.x; i < frameLength; i += blockDim.x)
				dst[i] = src[i];
			__syncthreads();

			if (threadIdx.x == 0) {
				int rem = codeLength % 8;
				if (rem != 0)
					dst[frameLength - 1] &= (uint8_t)((1u << rem) - 1u);

				d_pivotsMatrix[mrows] = rowIndex;
				d_matrixRows[0] = mrows + 1;
				d_rowType[sel] = -1;
			}
			__syncthreads();
			return;
		}

		// 2) редукция: pivot < rowIndex
		int cand = DevFindLessType(d_cwIndexFast, d_rowType, wordsCount, rowIndex, 0);
		if (cand == INT_MAX)
			cand = DevFindLessType(d_cwIndexFast, d_rowType, wordsCount, rowIndex, 1);

		if (cand != INT_MAX) {
			int currentPivot = d_cwIndexFast[cand];
			int mrows = d_matrixRows[0];

			__shared__ int target;
			if (threadIdx.x == 0) target = INT_MAX;
			__syncthreads();

			for (int i = threadIdx.x; i < mrows; i += blockDim.x) {
				if (d_pivotsMatrix[i] == currentPivot)
					atomicMin(&target, i);
			}
			__syncthreads();

			if (target != INT_MAX) {
				uint8_t* src = d_fastRows + cand * frameLength;
				const uint8_t* tgt = d_matrix + target * frameLength;

				DevXor(src, tgt, frameLength, codeLength);
				int newPivot = DevLeadingOne(src, codeLength);

				if (threadIdx.x == 0) {
					d_cwIndexFast[cand] = newPivot;
					if (newPivot == -1)
						d_rowType[cand] = -1;
					else if (newPivot > rowIndex)
						d_rowType[cand] = 1;
				}
				__syncthreads();
				continue;
			}
		}

		// 3) RAM с pivot > rowIndex → Cache
		__shared__ int moved;
		if (threadIdx.x == 0) moved = 0;
		__syncthreads();

		for (int i = threadIdx.x; i < wordsCount; i += blockDim.x) {
			if (d_rowType[i] == 0 && d_cwIndexFast[i] > rowIndex) {
				d_rowType[i] = 1;
				atomicExch(&moved, 1);
			}
		}
		__syncthreads();

		if (moved)
			continue;

		// 4) нулевая строка
		{
			int mrows = d_matrixRows[0];
			uint8_t* dst = d_matrix + mrows * frameLength;

			for (int i = threadIdx.x; i < frameLength; i += blockDim.x)
				dst[i] = 0;
			__syncthreads();

			if (threadIdx.x == 0) {
				d_pivotsMatrix[mrows] = -1;
				d_matrixRows[0] = mrows + 1;
			}
			__syncthreads();
			return;
		}
	}
}

// ============================================================================
// Запись результата
// ============================================================================

void WriteResultToFile(
	const string& fileName,
	const uint8_t* h_matrix,
	size_t matrixRows,
	size_t frameLength,
	size_t codeLength,
	bool isBis)
{
	ofstream out(fileName, ios::binary);
	if (!out.is_open()) {
		cerr << "Не удалось открыть файл для записи\n";
		return;
	}

	if (isBis) {
		size_t totalBytes = matrixRows * codeLength;
		uint8_t* buf = new uint8_t[totalBytes]();
		size_t pos = 0;
		for (size_t row = 0; row < matrixRows; ++row) {
			const uint8_t* rowPtr = h_matrix + row * frameLength;
			for (size_t b = 0; b < codeLength; ++b) {
				if (GetBitHost(rowPtr, b))
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
			const uint8_t* rowPtr = h_matrix + row * frameLength;
			for (size_t b = 0; b < codeLength; ++b) {
				if (GetBitHost(rowPtr, b))
					buf[outBitPos / 8] |= (1u << (outBitPos % 8));
				++outBitPos;
			}
		}
		out.write(reinterpret_cast<char*>(buf), totalBytes);
		delete[] buf;
	}
	out.close();
}

int main()
{
	setlocale(LC_ALL, "");
	auto start = chrono::high_resolution_clock::now();

	// Параметры
	const string InputFileName = "6.21.bin";
	const string TempDir = "";

	const size_t codeLength = 18004;
	const size_t infoLength = 16384;
	const size_t frameLength = (codeLength + 7) / 8;
	const char dataType = 0;   // 0 = BIN, !=0 = BIS

	// Открытие файла
	ifstream file(InputFileName, ios::binary);
	if (!file.is_open()) {
		cerr << "Ошибка открытия файла\n";
		return 1;
	}

	file.seekg(0, ios::end);
	size_t fileSizeBytes = file.tellg();
	file.seekg(0, ios::beg);

	size_t wordsCount = dataType
		? (fileSizeBytes / codeLength)
		: (fileSizeBytes * 8 / codeLength);

	size_t totalBytes = wordsCount * frameLength;

	// Чтение входных данных
	uint8_t* fileData = new uint8_t[fileSizeBytes];
	file.read(reinterpret_cast<char*>(fileData), fileSizeBytes);
	file.close();

	uint8_t* linearRows = nullptr;
	cudaMallocHost(&linearRows, totalBytes);

	for (size_t i = 0; i < wordsCount; ++i) {
		if (dataType)
			ReadCodeWordsBis(fileData, i, codeLength, linearRows + i * frameLength);
		else
			ReadCodeWordsBin(fileData, i, codeLength, linearRows + i * frameLength);
	}

	delete[] fileData;
	fileData = nullptr;

	uint8_t *d_fastRows = nullptr, *d_matrix = nullptr;
	int *d_cwIndexFast = nullptr, *d_rowType = nullptr;
	int *d_pivotsMatrix = nullptr, *d_matrixRows = nullptr;
	int *d_pivots = nullptr;

	cudaMalloc(&d_fastRows, totalBytes);
	cudaMemcpy(d_fastRows, linearRows, totalBytes, cudaMemcpyHostToDevice);
	cudaFreeHost(linearRows);   // если выделяли cudaMallocHost
	// если был new[] — delete[] linearRows;
	linearRows = nullptr;

	cudaMalloc(&d_matrix, infoLength * frameLength);
	cudaMemset(d_matrix, 0, infoLength * frameLength);

	cudaMalloc(&d_cwIndexFast, wordsCount * sizeof(int));
	cudaMalloc(&d_rowType, wordsCount * sizeof(int));
	cudaMalloc(&d_pivotsMatrix, infoLength * sizeof(int));
	cudaMalloc(&d_matrixRows, sizeof(int));
	cudaMalloc(&d_pivots, wordsCount * sizeof(int));

	cudaMemset(d_rowType, 0, wordsCount * sizeof(int));           // все RAM
	cudaMemset(d_pivotsMatrix, 0xFF, infoLength * sizeof(int));   // -1
	cudaMemset(d_matrixRows, 0, sizeof(int));

	// первичные пивоты → остаются на GPU
	{
		int warpsPerBlock = THREADS_PER_BLOCK / 32;
		int blocks = (int)((wordsCount + warpsPerBlock - 1) / warpsPerBlock);
		FindLeadingOneKernel << <blocks, THREADS_PER_BLOCK >> > (
			d_fastRows, d_pivots, (int)frameLength, (int)wordsCount);
		cudaDeviceSynchronize();

		cudaMemcpy(d_cwIndexFast, d_pivots, wordsCount * sizeof(int),
			cudaMemcpyDeviceToDevice);
		cudaFree(d_pivots);
		d_pivots = nullptr;
	}

	for (int rowIndex = 0; rowIndex < (int)infoLength; ++rowIndex) {
		ProcessOneRowKernel << <1, THREADS_PER_BLOCK >> > (
			rowIndex,
			d_fastRows,
			d_matrix,
			d_cwIndexFast,
			d_rowType,
			d_pivotsMatrix,
			d_matrixRows,
			(int)wordsCount,
			(int)frameLength,
			(int)codeLength);
	}
	cudaDeviceSynchronize();

	int matrixRows = 0;
	cudaMemcpy(&matrixRows, d_matrixRows, sizeof(int), cudaMemcpyDeviceToHost);

	if (matrixRows > 0) {
		ReverseGearGPU(d_matrix, (size_t)matrixRows, frameLength,
			d_pivotsMatrix, (int)codeLength);
	}

	uint8_t* h_matrix = new uint8_t[(size_t)matrixRows * frameLength]();
	if (matrixRows > 0) {
		cudaMemcpy(h_matrix, d_matrix,
			(size_t)matrixRows * frameLength, cudaMemcpyDeviceToHost);
	}

	string outName = TempDir + (dataType ? "result.bis" : "result.bin");
	WriteResultToFile(outName, h_matrix, (size_t)matrixRows, frameLength,
		codeLength, dataType != 0);

	delete[] h_matrix;

	cudaFree(d_fastRows);
	cudaFree(d_matrix);
	cudaFree(d_cwIndexFast);
	cudaFree(d_rowType);
	cudaFree(d_pivotsMatrix);
	cudaFree(d_matrixRows);

	auto ms = chrono::duration_cast<chrono::milliseconds>(
		chrono::high_resolution_clock::now() - start).count();

	cout << "Время выполнения программы в мс: " << ms << endl;
	return 0;
}