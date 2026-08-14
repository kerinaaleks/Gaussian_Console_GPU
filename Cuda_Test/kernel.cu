#include <iostream>
#include <fstream>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

using namespace std;
constexpr int THREADS_PER_BLOCK = 256;

// ============================================================================
// ЯДРА
// ============================================================================

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
	uint8_t* d_fastRows,
	uint8_t* d_cache,
	uint8_t* d_matrix,
	int frameLength,
	int srcIndex,
	bool fromCache,
	int targetRow,
	int codeLength)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= frameLength) return;

	uint8_t* src = fromCache
		? (d_cache + srcIndex * frameLength)
		: (d_fastRows + srcIndex * frameLength);
	uint8_t* target = d_matrix + targetRow * frameLength;

	src[tid] ^= target[tid];

	// очистка хвоста
	if (tid == frameLength - 1) {
		int rem = codeLength % 8;
		if (rem != 0)
			src[tid] &= (uint8_t)((1u << rem) - 1u);
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

__global__ void ShiftCacheKernel(uint8_t* d_cache, int frameLength, int start, int cacheRows)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= frameLength) return;
	for (int row = start; row < cacheRows - 1; ++row)
		d_cache[row * frameLength + tid] = d_cache[(row + 1) * frameLength + tid];
}

__global__ void ShiftIntKernel(int* arr, int start, int count)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid + start >= count - 1) return;
	arr[start + tid] = arr[start + tid + 1];
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

void ReadCodeWordsBin(ifstream& file, size_t cwIndex, size_t codeLength, uint8_t* dst) {
	size_t bitStart = cwIndex * codeLength;
	size_t byteStart = bitStart / 8;
	size_t bitOffset = bitStart % 8;
	size_t bytesToRead = (bitOffset + codeLength + 7) / 8;
	size_t frameLength = (codeLength + 7) / 8;

	uint8_t* buf = new uint8_t[bytesToRead];
	file.seekg(byteStart);
	file.read(reinterpret_cast<char*>(buf), bytesToRead);

	memset(dst, 0, frameLength);

	for (size_t i = 0; i < codeLength; ++i) {
		size_t srcByte = (bitOffset + i) / 8;
		size_t srcBit = (bitOffset + i) % 8;
		if ((buf[srcByte] >> srcBit) & 1) {
			size_t dstByte = i / 8;
			size_t dstBit = i % 8;
			dst[dstByte] |= (1u << dstBit);
		}
	}

	// очистка хвоста
	size_t rem = codeLength % 8;
	if (rem != 0)
		dst[frameLength - 1] &= (uint8_t)((1u << rem) - 1u);

	delete[] buf;
}

void ReadCodeWordsBis(ifstream& file, size_t cwIndex, size_t codeLength, uint8_t* dst)
{
	size_t frameLength = (codeLength + 7) / 8;
	size_t byteStart = cwIndex * codeLength;

	uint8_t* buf = new uint8_t[codeLength];
	file.seekg(byteStart);
	file.read(reinterpret_cast<char*>(buf), codeLength);

	memset(dst, 0, frameLength);

	for (size_t i = 0; i < codeLength; ++i) {
		if (buf[i] != 0) {
			size_t dstByte = i / 8;
			size_t dstBit = i % 8;
			dst[dstByte] |= (1u << dstBit);
		}
	}
	size_t rem = codeLength % 8;
	if (rem != 0)
		dst[frameLength - 1] &= (uint8_t)((1u << rem) - 1u);

	delete[] buf;
}

// ============================================================================
// Основная логика Гаусса
// ============================================================================

bool ProcessFrameGPU(
	int srcIndex, bool fromCache,
	uint8_t* d_fastRows, uint8_t* d_cache, uint8_t* d_matrix,
	uint8_t* frameBuffer, uint8_t** matrix,
	size_t codeLength, size_t frameLength,
	size_t& matrixRows, size_t& cacheRows,
	int rowIndex, size_t wordsCount, int pivot,
	int* cwIndexFast, int* cwIndexCache, int* pivotsMatrix,
	int* d_pivotsMatrix, int* d_foundRow, int* d_pivotOut, int* d_cwIndexCache)
{
	if (pivot == -1) {
		if (fromCache) {
			cwIndexCache[srcIndex] = -1;
			int m1 = -1;
			cudaMemcpy(d_cwIndexCache + srcIndex, &m1, sizeof(int), cudaMemcpyHostToDevice);
		}
		else {
			cwIndexFast[srcIndex] = -1;
		}
		return false;
	}

	while (true) {
		if (pivot == rowIndex) {
			memcpy(matrix[rowIndex], frameBuffer, frameLength);
			cudaMemcpy(d_matrix + rowIndex * frameLength, frameBuffer, frameLength, cudaMemcpyHostToDevice);
			pivotsMatrix[rowIndex] = pivot;
			matrixRows++;
			cudaMemcpy(d_pivotsMatrix + rowIndex, &pivot, sizeof(int), cudaMemcpyHostToDevice);
			return true;
		}

		if (pivot < rowIndex) {
			int init = INT_MAX;
			cudaMemcpy(d_foundRow, &init, sizeof(int), cudaMemcpyHostToDevice);

			int blocks = (int)((matrixRows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
			FindPivotKernel << <blocks, THREADS_PER_BLOCK >> > (d_pivotsMatrix, (int)matrixRows, pivot, d_foundRow);
			cudaDeviceSynchronize();

			int target_row;
			cudaMemcpy(&target_row, d_foundRow, sizeof(int), cudaMemcpyDeviceToHost);
			if (target_row == INT_MAX) target_row = -1;
			if (target_row == -1) return false;

			int blocksXor = (int)((frameLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
			XorRowsKernel << <blocksXor, THREADS_PER_BLOCK >> > (
				d_fastRows, d_cache, d_matrix,
				(int)frameLength, srcIndex, fromCache, target_row, (int)codeLength);
			cudaDeviceSynchronize();

			uint8_t* srcPtr = fromCache
				? (d_cache + srcIndex * frameLength)
				: (d_fastRows + srcIndex * frameLength);

			FindPivotInSingleRow << <1, THREADS_PER_BLOCK >> > (srcPtr, (int)frameLength, d_pivotOut);
			cudaDeviceSynchronize();

			cudaMemcpy(frameBuffer, srcPtr, frameLength, cudaMemcpyDeviceToHost);
			cudaMemcpy(&pivot, d_pivotOut, sizeof(int), cudaMemcpyDeviceToHost);

			if (fromCache) {
				cwIndexCache[srcIndex] = pivot;
				cudaMemcpy(d_cwIndexCache + srcIndex, &pivot, sizeof(int), cudaMemcpyHostToDevice);
			}
			else {
				cwIndexFast[srcIndex] = pivot;
			}

			if (pivot == -1) {
				if (fromCache) {
					cwIndexCache[srcIndex] = -1;
					int m1 = -1;
					cudaMemcpy(d_cwIndexCache + srcIndex, &m1, sizeof(int), cudaMemcpyHostToDevice);
				}
				else {
					cwIndexFast[srcIndex] = -1;
				}
				return false;
			}
			continue;
		}

		// pivot > rowIndex
		if (fromCache) {
			cwIndexCache[srcIndex] = pivot;
			cudaMemcpy(d_cwIndexCache + srcIndex, &pivot, sizeof(int), cudaMemcpyHostToDevice);
			return false;
		}
		else {
			if (cacheRows < wordsCount) {
				cudaMemcpy(d_cache + cacheRows * frameLength, frameBuffer, frameLength, cudaMemcpyHostToDevice);
				cwIndexCache[cacheRows] = pivot;
				cudaMemcpy(d_cwIndexCache + cacheRows, &pivot, sizeof(int), cudaMemcpyHostToDevice);
				cacheRows++;
			}
			return false;
		}
	}
}

void ProcessAllFrame(
	uint8_t** matrix, uint8_t* frameBuffer,
	uint8_t* d_fastRows, uint8_t* d_cache, uint8_t* d_matrix,
	size_t wordsCount, size_t codeLength, size_t frameLength,
	size_t& matrixRows, size_t& cacheRows, size_t infoLength,
	int* cwIndexFast, int* cwIndexCache, int* pivotsMatrix,
	int* d_pivotsMatrix, int* d_foundRow, int* d_cwIndexCache, int* d_pivotOut)
{
	for (int rowIndex = 0; rowIndex < (int)infoLength; ++rowIndex) {
		bool found = false;

		// 1. Поиск в fastRows
		for (size_t i = 0; i < wordsCount && !found; ++i) {
			if (cwIndexFast[i] == -1) continue;

			cudaMemcpy(frameBuffer, d_fastRows + i * frameLength, frameLength, cudaMemcpyDeviceToHost);
			int pivot = cwIndexFast[i];

			if (ProcessFrameGPU(
				(int)i, false, d_fastRows, d_cache, d_matrix, frameBuffer, matrix,
				codeLength, frameLength, matrixRows, cacheRows, rowIndex, wordsCount,
				pivot, cwIndexFast, cwIndexCache, pivotsMatrix,
				d_pivotsMatrix, d_foundRow, d_pivotOut, d_cwIndexCache))
			{
				found = true;
				cwIndexFast[i] = -1;

				// подтягиваем строку из кэша на освободившееся место
				size_t j = 0;
				while (j < cacheRows) {
					if (cwIndexCache[j] == -1) {
						// просто сдвигаем
						int blocks = (int)((frameLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
						ShiftCacheKernel << <blocks, THREADS_PER_BLOCK >> > (d_cache, (int)frameLength, (int)j, (int)cacheRows);
						ShiftIntKernel << <(cacheRows - j + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_cwIndexCache, (int)j, (int)cacheRows);
						cudaDeviceSynchronize();

						for (size_t k = j + 1; k < cacheRows; ++k)
							cwIndexCache[k - 1] = cwIndexCache[k];
						cacheRows--;
						cwIndexCache[cacheRows] = -1;
						continue;
					}

					cudaMemcpy(d_fastRows + i * frameLength,
						d_cache + j * frameLength,
						frameLength, cudaMemcpyDeviceToDevice);
					cwIndexFast[i] = cwIndexCache[j];

					int blocks = (int)((frameLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
					ShiftCacheKernel << <blocks, THREADS_PER_BLOCK >> > (d_cache, (int)frameLength, (int)j, (int)cacheRows);
					ShiftIntKernel << <(cacheRows - j + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_cwIndexCache, (int)j, (int)cacheRows);
					cudaDeviceSynchronize();

					for (size_t k = j + 1; k < cacheRows; ++k)
						cwIndexCache[k - 1] = cwIndexCache[k];
					cacheRows--;
					cwIndexCache[cacheRows] = -1;
					break;
				}
			}
		}

		// 2. Поиск в кэше
		if (!found && cacheRows > 0) {
			int init = INT_MAX;
			cudaMemcpy(d_foundRow, &init, sizeof(int), cudaMemcpyHostToDevice);

			int blocks = (int)((cacheRows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
			FindPivotKernel << <blocks, THREADS_PER_BLOCK >> > (d_cwIndexCache, (int)cacheRows, rowIndex, d_foundRow);
			cudaDeviceSynchronize();

			int target_row;
			cudaMemcpy(&target_row, d_foundRow, sizeof(int), cudaMemcpyDeviceToHost);

			if (target_row != INT_MAX) {
				cudaMemcpy(frameBuffer, d_cache + target_row * frameLength, frameLength, cudaMemcpyDeviceToHost);
				int pivot = cwIndexCache[target_row];

				if (ProcessFrameGPU(
					target_row, true, d_fastRows, d_cache, d_matrix, frameBuffer, matrix,
					codeLength, frameLength, matrixRows, cacheRows, rowIndex, wordsCount,
					pivot, cwIndexFast, cwIndexCache, pivotsMatrix,
					d_pivotsMatrix, d_foundRow, d_pivotOut, d_cwIndexCache))
				{
					int blocks = (int)((frameLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
					ShiftCacheKernel << <blocks, THREADS_PER_BLOCK >> > (d_cache, (int)frameLength, target_row, (int)cacheRows);
					ShiftIntKernel << <(cacheRows - target_row + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_cwIndexCache, target_row, (int)cacheRows);
					cudaDeviceSynchronize();

					for (size_t j = target_row + 1; j < cacheRows; ++j)
						cwIndexCache[j - 1] = cwIndexCache[j];
					cacheRows--;
					cwIndexCache[cacheRows] = -1;
					found = true;
				}
			}
		}

		// 3. Нулевая строка
		if (!found) {
			memset(matrix[rowIndex], 0, frameLength);
			pivotsMatrix[rowIndex] = -1;
			matrixRows++;
			int m1 = -1;
			cudaMemcpy(d_pivotsMatrix + rowIndex, &m1, sizeof(int), cudaMemcpyHostToDevice);
		}
	}

	// Обратный ход
	ReverseGearGPU(d_matrix, matrixRows, frameLength, d_pivotsMatrix, (int)codeLength);

	for (size_t i = 0; i < infoLength; ++i)
		cudaMemcpy(matrix[i], d_matrix + i * frameLength, frameLength, cudaMemcpyDeviceToHost);
}

// ============================================================================
// Запись результата (BIN)
// ============================================================================

void WriteResultToFile(
	const string& fileName,
	uint8_t** matrix,
	size_t matrixRows,
	size_t codeLength,
	bool isBis) //
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
			for (size_t b = 0; b < codeLength; ++b) {
				if (GetBitHost(matrix[row], b))
					buf[pos] = 0xFF;

				++pos;
			}
		}
		out.write(reinterpret_cast<char*> (buf), totalBytes);
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
					size_t byteIndex = outBitPos / 8;
					size_t bitIndex = outBitPos % 8;
					buf[byteIndex] |= (1u << bitIndex);
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

	// ========================================================================
	// Параметры
	// ========================================================================
	const string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\11.1.bis)";
	const string TempDir = R"(D:\Rubin\sessions\tmp_1783328386069\)";

	const size_t codeLength = 2000;
	const size_t infoLength = 2000;
	const size_t frameLength = (codeLength + 7) / 8;

	const char dataType = 1;   // 0 = bin, 1 (или любое ненулевое) = bis

	// ========================================================================
	// Открытие файла
	// ========================================================================
	ifstream file(InputFileName, ios::binary);
	if (!file.is_open()) {
		cerr << "Ошибка открытия файла\n";
		return 1;
	}

	file.seekg(0, ios::end);
	size_t fileSizeBytes = file.tellg();
	file.seekg(0, ios::beg);

	// Количество кодовых слов зависит от формата
	size_t wordsCount = dataType
		? (fileSizeBytes / codeLength) // BIS: 1 байт = 1 бит
		: (fileSizeBytes * 8 / codeLength); // BIN: упакованные биты

	size_t totalBytes = wordsCount * frameLength;

	// ========================================================================
	// Матрица результата
	// ========================================================================
	uint8_t** matrix = new uint8_t*[infoLength];
	for (size_t i = 0; i < infoLength; ++i)
		matrix[i] = new uint8_t[frameLength]();

	// ========================================================================
	// Чтение входных данных
	// ========================================================================
	uint8_t* linearRows = new uint8_t[totalBytes];
	for (size_t i = 0; i < wordsCount; ++i) {
		if (dataType)
			ReadCodeWordsBis(file, i, codeLength, linearRows + i * frameLength);
		else
			ReadCodeWordsBin(file, i, codeLength, linearRows + i * frameLength);
	}
	file.close();

	// ========================================================================
	// Индексы
	// ========================================================================
	int* cwIndexFast = new int[wordsCount];
	int* cwIndexCache = new int[wordsCount];
	for (size_t i = 0; i < wordsCount; ++i) {
		cwIndexFast[i] = -1;
		cwIndexCache[i] = -1;
	}

	int* pivotsMatrix = new int[infoLength];
	for (size_t i = 0; i < infoLength; ++i)
		pivotsMatrix[i] = -1;

	// ========================================================================
	// GPU-память
	// ========================================================================
	uint8_t *d_fastRows = nullptr, *d_cache = nullptr, *d_matrix = nullptr;
	int *d_pivots = nullptr, *d_pivotsMatrix = nullptr;
	int *d_foundRow = nullptr, *d_cwIndexCache = nullptr, *d_pivotOut = nullptr;

	cudaMalloc(&d_fastRows, totalBytes);
	cudaMemcpy(d_fastRows, linearRows, totalBytes, cudaMemcpyHostToDevice);
	delete[] linearRows;

	cudaMalloc(&d_cache, wordsCount * frameLength);
	cudaMemset(d_cache, 0, wordsCount * frameLength);

	cudaMalloc(&d_matrix, infoLength * frameLength);
	cudaMemset(d_matrix, 0, infoLength * frameLength);

	cudaMalloc(&d_pivots, wordsCount * sizeof(int));

	// ========================================================================
	// Первичный поиск ведущих единиц
	// ========================================================================
	{
		int warpsPerBlock = THREADS_PER_BLOCK / 32;
		int blocks = (int)((wordsCount + warpsPerBlock - 1) / warpsPerBlock);
		FindLeadingOneKernel << <blocks, THREADS_PER_BLOCK >> > (
			d_fastRows, d_pivots, (int)frameLength, (int)wordsCount);
		cudaDeviceSynchronize();
		cudaMemcpy(cwIndexFast, d_pivots, wordsCount * sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(d_pivots);
		d_pivots = nullptr;
	}

	cudaMalloc(&d_pivotsMatrix, infoLength * sizeof(int));
	cudaMemcpy(d_pivotsMatrix, pivotsMatrix, infoLength * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_foundRow, sizeof(int));
	cudaMalloc(&d_cwIndexCache, wordsCount * sizeof(int));
	cudaMemcpy(d_cwIndexCache, cwIndexCache, wordsCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&d_pivotOut, sizeof(int));

	// ========================================================================
	// Гаусс
	// ========================================================================
	uint8_t* frameBuffer = new uint8_t[frameLength];
	size_t matrixRows = 0, cacheRows = 0;

	ProcessAllFrame(
		matrix, frameBuffer,
		d_fastRows, d_cache, d_matrix,
		wordsCount, codeLength, frameLength,
		matrixRows, cacheRows, infoLength,
		cwIndexFast, cwIndexCache, pivotsMatrix,
		d_pivotsMatrix, d_foundRow, d_cwIndexCache, d_pivotOut);

	// ========================================================================
	// Запись результата в том же формате
	// ========================================================================
	string outName = TempDir + (dataType ? "result.bis" : "result.bin");
	WriteResultToFile(outName, matrix, matrixRows, codeLength, dataType != 0);

	// ========================================================================
	// Очистка
	// ========================================================================
	for (size_t i = 0; i < infoLength; ++i) delete[] matrix[i];
	delete[] matrix;
	delete[] frameBuffer;
	delete[] cwIndexFast;
	delete[] cwIndexCache;
	delete[] pivotsMatrix;

	cudaFree(d_fastRows);
	cudaFree(d_cache);
	cudaFree(d_matrix);
	cudaFree(d_pivotsMatrix);
	cudaFree(d_foundRow);
	cudaFree(d_cwIndexCache);
	cudaFree(d_pivotOut);

	auto ms = chrono::duration_cast<chrono::milliseconds>(
		chrono::high_resolution_clock::now() - start).count();
	cout << "Время выполнения программы в мс: " << ms << endl;

	return 0;
}