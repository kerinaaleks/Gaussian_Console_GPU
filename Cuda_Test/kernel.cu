#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

using namespace std;

// Точные копии из эталона
inline bool GetBit(char *data, int bitNum)
{
	int byteNum = bitNum / 8;
	int shift = bitNum % 8;
	return (data[byteNum] & (0x01 << shift)) != 0;
}

inline void SetBit(char *data, int bitNum, bool bitVal)
{
	int byteNum = bitNum / 8;
	int shift = bitNum % 8;
	if (bitVal)
		data[byteNum] |= (0x01 << shift);
	else
		data[byteNum] &= ~(0x01 << shift);
}

// ============================================================================
// ЯДРА (Логика пивотов из первого файла)
// ============================================================================

__global__ void FindLeadingOneKernel(const uint8_t* __restrict__ rows, int* __restrict__ pivots, int frameLength, int wordsCount) {
	int laneId = threadIdx.x & 31;
	int warpId = threadIdx.x >> 5;
	int warpsPerBlock = blockDim.x >> 5;
	int rowIndex = blockIdx.x * warpsPerBlock + warpId;

	if (rowIndex >= wordsCount)
		return;

	const uint8_t* row = rows + rowIndex * frameLength;
	int localPivot = INT_MAX;

	// Покрываем ВСЮ строку (логика из 1-го файла)
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
		int other = __shfl_down_sync(0xFFFFFFFF, localPivot, offset);
		localPivot = min(localPivot, other);
	}

	if (laneId == 0) {
		pivots[rowIndex] = (localPivot == INT_MAX ? -1 : localPivot);
	}
}

__global__ void FindPivotInSingleRow(const uint8_t* row, int frameLength, int* pivotOut) {
	int tid = threadIdx.x;
	int laneId = threadIdx.x & 31;
	int warpId = threadIdx.x >> 5;

	__shared__ int warpMin[8];

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

	for (int offset = 16; offset > 0; offset >>= 1) {
		int other = __shfl_down_sync(0xFFFFFFFF, localPivot, offset);
		localPivot = min(localPivot, other);
	}

	if (laneId == 0)
		warpMin[warpId] = localPivot;

	__syncthreads();

	if (tid == 0) {
		int finalPivot = INT_MAX;
		for (int i = 0; i < (blockDim.x >> 5); ++i)
			finalPivot = min(finalPivot, warpMin[i]);
		*pivotOut = (finalPivot == INT_MAX ? -1 : finalPivot);
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
	int codeLength
) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= frameLength) return;

	uint8_t* src = fromCache ? (d_cache + srcIndex * frameLength)
		: (d_fastRows + srcIndex * frameLength);
	uint8_t* target = d_matrix + targetRow * frameLength;

	src[tid] ^= target[tid];

	// Если это самый последний байт строки, зануляем биты за пределами codeLength
	if (tid == frameLength - 1) {
		int rem = codeLength % 8;
		if (rem != 0) {
			uint8_t mask = (1 << rem) - 1;
			src[tid] &= mask;
		}
	}
}

__device__ __forceinline__ bool GetBitDevice(const uint8_t* row, int bitIndex) {
	int byte = bitIndex / 8;
	int bit = bitIndex % 8;
	return (row[byte] >> bit) & 1;
}

inline bool GetBitHost(const uint8_t* row, size_t bitIndex) {
	size_t byte = bitIndex / 8;
	size_t bit = bitIndex % 8;
	return (row[byte] >> bit) & 1;
}

// Обратный ход
__global__ void ReverseGearKernel(
	uint8_t* d_matrix,
	int matrixRows,
	int frameLength,
	int pivotRow,
	int* d_pivotsMatrix,
	int codeLength) // Добавили codeLength в аргументы
{
	int j = blockIdx.x;
	if (j >= matrixRows) return;
	if (j == pivotRow) return;

	uint8_t* rowJ = d_matrix + j * frameLength;
	const uint8_t* rowI = d_matrix + pivotRow * frameLength;
	int pivotColumn = d_pivotsMatrix[pivotRow];
	if (pivotColumn < 0) return;
	if (!GetBitDevice(rowJ, pivotColumn)) return;

	int tid = threadIdx.x;
	for (int idx = tid; idx < frameLength; idx += blockDim.x) {
		rowJ[idx] ^= rowI[idx];
		if (idx == frameLength - 1) {
			int rem = codeLength % 8;
			if (rem != 0) {
				uint8_t mask = (1 << rem) - 1;
				rowJ[idx] &= mask;
			}
		}
	}
}

__global__ void FindPivotKernel(const int* pivotsMatrix, int matrixRows, int pivot, int* result) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= matrixRows) return;
	if (pivotsMatrix[tid] == pivot)
		atomicMin(result, tid);
}

__global__ void ShiftCacheKernel(uint8_t* d_cache, int frameLength, int start, int cacheRows) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= frameLength) return;
	for (int row = start; row < cacheRows - 1; row++)
		d_cache[row * frameLength + tid] = d_cache[(row + 1) * frameLength + tid];
}

__global__ void ShiftIntKernel(int* arr, int start, int count) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid + start >= count - 1) return;
	arr[start + tid] = arr[start + tid + 1];
}

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ CPU
// ============================================================================

void ReverseGearGPU(uint8_t* d_matrix, size_t matrixRows, size_t frameLength, int* d_pivotsMatrix, int codeLength) {
	int threadPerBlock = 256;
	for (int i = (int)matrixRows - 1; i >= 0; --i) {
		int blocks = i;
		if (blocks > 0)
			ReverseGearKernel << <blocks, threadPerBlock >> > (d_matrix, (int)matrixRows, (int)frameLength, i, d_pivotsMatrix, codeLength);
	}
	cudaDeviceSynchronize();
}

void ReadCodeWords(ifstream& file, size_t cwIndex, size_t codeLength, uint8_t* dst) {
	size_t bitStart = cwIndex * codeLength;
	size_t byteStart = bitStart / 8;
	size_t bitOffset = bitStart % 8;
	size_t bytesToRead = (bitOffset + codeLength + 7) / 8;

	uint8_t* buf = new uint8_t[bytesToRead];
	file.seekg(byteStart);
	file.read((char*)buf, bytesToRead);

	size_t frameLength = (codeLength + 7) / 8;
	memset(dst, 0, frameLength);

	for (size_t i = 0; i < codeLength; i++) {
		size_t srcByte = (bitOffset + i) / 8;
		size_t srcBit = (bitOffset + i) % 8;
		bool bit = (buf[srcByte] >> srcBit) & 1;
		if (bit) {
			size_t dstByte = i / 8;
			size_t dstBit = i % 8;
			dst[dstByte] |= (1 << dstBit);
		}
	}

	// Очищаем хвостик последнего байта, если длина не кратна 8
	size_t rem = codeLength % 8;
	if (rem != 0) {
		uint8_t mask = (1 << rem) - 1;
		dst[frameLength - 1] &= mask;
	}

	delete[] buf;
}

// ============================================================================
// ОСНОВНАЯ ЛОГИКА
// ============================================================================

bool ProcessFrameGPU(
	int srcIndex,
	bool fromCache,
	uint8_t* d_fastRows,
	uint8_t* d_cache,
	uint8_t* d_matrix,
	uint8_t* frameBuffer,
	uint8_t** matrix,
	size_t codeLength,
	size_t frameLength,
	size_t& matrixRows,
	size_t& cacheRows,
	int rowIndex,
	size_t wordsCount,
	int pivot,
	int* cwIndexFast,
	int* cwIndexCache,
	int* pivotsMatrix,
	int* d_pivotsMatrix,
	int* d_foundRow,
	int* d_pivotOut,
	int* d_cwIndexCache
)
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

	while (true)
	{
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

			int threads = 256;
			int blocks = (int)((matrixRows + threads - 1) / threads);
			FindPivotKernel << <blocks, threads >> > (d_pivotsMatrix, (int)matrixRows, pivot, d_foundRow);
			cudaDeviceSynchronize();

			int target_row;
			cudaMemcpy(&target_row, d_foundRow, sizeof(int), cudaMemcpyDeviceToHost);
			if (target_row == INT_MAX) target_row = -1;
			if (target_row == -1)
				return false;

			int threadsPerBlock = 256;
			int blocksXor = (int)((frameLength + threadsPerBlock - 1) / threadsPerBlock);
			XorRowsKernel << <blocksXor, threadsPerBlock >> > (
				d_fastRows, d_cache, d_matrix, (int)frameLength, srcIndex, fromCache, target_row, (int)codeLength);
			cudaDeviceSynchronize();

			uint8_t* srcPtr = fromCache ? (d_cache + srcIndex * frameLength)
				: (d_fastRows + srcIndex * frameLength);
			FindPivotInSingleRow << <1, 256 >> > (srcPtr, (int)frameLength, d_pivotOut);
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

		if (pivot > rowIndex) {
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
}

void ProcessAllFrame(
	uint8_t** matrix,
	uint8_t* frameBuffer,
	uint8_t* d_fastRows,
	uint8_t* d_cache,
	uint8_t* d_matrix,
	size_t wordsCount,
	size_t codeLength,
	size_t frameLength,
	size_t& matrixRows,
	size_t& cacheRows,
	size_t infoLength,
	int* cwIndexFast,
	int* cwIndexCache,
	int* pivotsMatrix,
	int* d_pivotsMatrix,
	int* d_foundRow,
	int* d_cwIndexCache,
	int* d_pivotOut
)
{
	for (int rowIndex = 0; rowIndex < (int)infoLength; ++rowIndex) {
		bool found = false;

		// 1. Ищем строку в RAM
		for (size_t i = 0; i < wordsCount && !found; ++i) {
			if (cwIndexFast[i] == -1)
				continue;

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

				// Заменяем освободившуюся RAM строкой из кэша
				size_t j = 0;
				while (j < cacheRows) {
					if (cwIndexCache[j] == -1) {
						int threadsShift = 256;
						int blocksShift = (int)((frameLength + threadsShift - 1) / threadsShift);
						ShiftCacheKernel << <blocksShift, threadsShift >> > (d_cache, (int)frameLength, (int)j, (int)cacheRows);
						int blocksInt = (int)(((cacheRows - 1 - j) + threadsShift - 1) / threadsShift);
						ShiftIntKernel << <blocksInt, threadsShift >> > (d_cwIndexCache, (int)j, (int)cacheRows);
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

					int threadsShift = 256;
					int blocksShift = (int)((frameLength + threadsShift - 1) / threadsShift);
					ShiftCacheKernel << <blocksShift, threadsShift >> > (d_cache, (int)frameLength, (int)j, (int)cacheRows);
					int blocksInt = (int)(((cacheRows - 1 - j) + threadsShift - 1) / threadsShift);
					ShiftIntKernel << <blocksInt, threadsShift >> > (d_cwIndexCache, (int)j, (int)cacheRows);
					cudaDeviceSynchronize();

					for (size_t k = j + 1; k < cacheRows; ++k)
						cwIndexCache[k - 1] = cwIndexCache[k];

					cacheRows--;
					cwIndexCache[cacheRows] = -1;
					break;
				}
			}
		}

		// 2. Ищем в кэше
		if (!found && cacheRows > 0) {
			int init = INT_MAX;
			cudaMemcpy(d_foundRow, &init, sizeof(int), cudaMemcpyHostToDevice);

			int threads = 256;
			int blocks = (int)((cacheRows + threads - 1) / threads);
			FindPivotKernel << <blocks, threads >> > (d_cwIndexCache, (int)cacheRows, rowIndex, d_foundRow);
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
					int threadsShift = 256;
					int blocksShift = (int)((frameLength + threadsShift - 1) / threadsShift);
					ShiftCacheKernel << <blocksShift, threadsShift >> > (d_cache, (int)frameLength, target_row, (int)cacheRows);

					int countShift = (int)cacheRows - 1;
					int blocksInt = (int)(((countShift - target_row) + threadsShift - 1) / threadsShift);
					ShiftIntKernel << <blocksInt, threadsShift >> > (d_cwIndexCache, target_row, (int)cacheRows);
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

	// 4. Обратный ход
	ReverseGearGPU(d_matrix, matrixRows, frameLength, d_pivotsMatrix, (int)codeLength);

	for (size_t i = 0; i < infoLength; ++i) {
		cudaMemcpy(matrix[i], d_matrix + i * frameLength, frameLength, cudaMemcpyDeviceToHost);
	}
}

// ============================================================================
// ВВОД/ВЫВОД
// ============================================================================

void WritePackedBitsToFile(
	const std::string& fileName,
	uint8_t** matrix,
	size_t matrixRows,
	size_t codeLength)
{
	std::ofstream out(fileName, std::ios::binary);
	if (!out.is_open()) {
		std::cout << "Не удалось открыть файл для записи\n";
		return;
	}

	// Сколько всего бит и байт нужно
	size_t totalBits = matrixRows * codeLength;
	size_t totalBytes = (totalBits + 7) / 8;

	uint8_t* buf = new uint8_t[totalBytes];
	memset(buf, 0, totalBytes);

	size_t outBitPos = 0;

	for (size_t row = 0; row < matrixRows; ++row)
	{
		uint8_t* data = matrix[row];

		// Берём ровно codeLength бит (хвост автоматически игнорируется)
		for (size_t b = 0; b < codeLength; ++b)
		{
			bool bit = GetBitHost(data, b);   // или GetBit((char*)data, b)

			if (bit)
			{
				size_t byteIndex = outBitPos / 8;
				size_t bitIndex = outBitPos % 8;
				buf[byteIndex] |= (1u << bitIndex);   // LSB-first
			}
			outBitPos++;
		}
	}

	out.write(reinterpret_cast<char*>(buf), totalBytes);
	out.close();

	delete[] buf;
}

// ============================================================================
// MAIN
// ============================================================================

int main()
{
	setlocale(LC_ALL, "");

	auto start = chrono::high_resolution_clock::now();

	string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\6.21.bin)";
	string TempDir = R"(D:\Rubin\sessions\tmp_1783328386069\)";
	size_t codeLength = 18004;
	size_t infoLength = 16384;

	size_t frameLength = (codeLength + 7) / 8;
	uint8_t* frameBuffer = new uint8_t[frameLength];

	size_t matrixRows = 0;
	size_t cacheRows = 0;

	ifstream file(InputFileName, ios::binary);
	if (!file.is_open()) {
		cout << "Ошибка открытия файла\n";
		return 1;
	}
	cout << "Файл открыт\n";

	file.seekg(0, ios::end);
	size_t fileSizeBytes = file.tellg();
	file.seekg(0, ios::beg);

	size_t fileSizeBits = fileSizeBytes * 8;
	size_t wordsCount = fileSizeBits / codeLength;

	uint8_t** matrix = new uint8_t*[infoLength];
	for (size_t i = 0; i < infoLength; i++) {
		matrix[i] = new uint8_t[frameLength];
		memset(matrix[i], 0, frameLength);
	}

	uint8_t** fastRows = new uint8_t*[wordsCount];
	for (size_t i = 0; i < wordsCount; i++) {
		fastRows[i] = new uint8_t[frameLength];
		ReadCodeWords(file, i, codeLength, fastRows[i]);
	}

	size_t totalBytes = wordsCount * frameLength;
	uint8_t* linearRows = new uint8_t[totalBytes];
	for (size_t i = 0; i < wordsCount; i++)
		memcpy(linearRows + i * frameLength, fastRows[i], frameLength);

	int* cwIndexFast = new int[wordsCount];
	int* cwIndexCache = new int[wordsCount];
	for (size_t i = 0; i < wordsCount; ++i) {
		cwIndexFast[i] = -1;
		cwIndexCache[i] = -1;
	}

	// ---------------- GPU ALLOC ----------------

	uint8_t* d_fastRows;
	cudaMalloc(&d_fastRows, totalBytes);
	cudaMemcpy(d_fastRows, linearRows, totalBytes, cudaMemcpyHostToDevice);

	uint8_t* d_cache;
	cudaMalloc(&d_cache, wordsCount * frameLength);
	cudaMemset(d_cache, 0, wordsCount * frameLength);

	uint8_t* d_matrix;
	cudaMalloc(&d_matrix, infoLength * frameLength);
	cudaMemset(d_matrix, 0, infoLength * frameLength);

	int* d_pivots;
	cudaMalloc(&d_pivots, wordsCount * sizeof(int));

	int* pivotsMatrix = new int[infoLength];
	for (size_t i = 0; i < infoLength; ++i)
		pivotsMatrix[i] = -1;

	// ---------------- GPU: FindLeadingOneKernel ----------------

	int threadsPerBlock = 256;
	int warpsPerBlock = threadsPerBlock / 32;
	int blocks = (int)((wordsCount + warpsPerBlock - 1) / warpsPerBlock);

	FindLeadingOneKernel << <blocks, threadsPerBlock >> > (
		d_fastRows, d_pivots, (int)frameLength, (int)wordsCount);
	cudaDeviceSynchronize();

	int* pivots = new int[wordsCount];
	cudaMemcpy(pivots, d_pivots, wordsCount * sizeof(int), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < wordsCount; ++i)
		cwIndexFast[i] = pivots[i];

	// ---------------- GPU: прочие буферы ----------------
	int* d_pivotsMatrix;
	cudaMalloc(&d_pivotsMatrix, infoLength * sizeof(int));
	cudaMemcpy(d_pivotsMatrix, pivotsMatrix, infoLength * sizeof(int), cudaMemcpyHostToDevice);

	int* d_foundRow;
	cudaMalloc(&d_foundRow, sizeof(int));

	int* d_cwIndexCache;
	cudaMalloc(&d_cwIndexCache, wordsCount * sizeof(int));
	cudaMemcpy(d_cwIndexCache, cwIndexCache, wordsCount * sizeof(int), cudaMemcpyHostToDevice);

	int* d_pivotOut;
	cudaMalloc(&d_pivotOut, sizeof(int));

	// ---------------- Гаусс ----------------
	ProcessAllFrame(
		matrix,
		frameBuffer,
		d_fastRows,
		d_cache,
		d_matrix,
		wordsCount,
		codeLength,
		frameLength,
		matrixRows,
		cacheRows,
		infoLength,
		cwIndexFast,
		cwIndexCache,
		pivotsMatrix,
		d_pivotsMatrix,
		d_foundRow,
		d_cwIndexCache,
		d_pivotOut);


	WritePackedBitsToFile(TempDir + "result.bin", matrix, matrixRows, codeLength);

	// ---------------- Очистка ----------------
	file.close();

	for (size_t i = 0; i < infoLength; i++) delete[] matrix[i];
	delete[] matrix;

	for (size_t i = 0; i < wordsCount; i++) delete[] fastRows[i];
	delete[] fastRows;

	delete[] frameBuffer;
	delete[] cwIndexFast;
	delete[] cwIndexCache;
	delete[] pivotsMatrix;
	delete[] pivots;
	delete[] linearRows;

	cudaFree(d_pivots);
	cudaFree(d_fastRows);
	cudaFree(d_cache);
	cudaFree(d_matrix);
	cudaFree(d_pivotsMatrix);
	cudaFree(d_foundRow);
	cudaFree(d_cwIndexCache);
	cudaFree(d_pivotOut);

	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
	cout << "Время выполнения программы в мс: " << duration.count() << endl;

	return 0;
}