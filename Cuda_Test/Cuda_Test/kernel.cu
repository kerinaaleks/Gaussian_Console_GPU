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
	uint8_t* d_matrix,
	int frameLength,
	int srcIndex,
	int targetRow,
	int codeLength)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= frameLength) return;

	uint8_t* src = d_fastRows + srcIndex * frameLength;
	uint8_t* target = d_matrix + targetRow * frameLength;

	src[tid] ^= target[tid];

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

// ============================================================================
// Основная логика Гаусса (в одном массиве с разделением приоритетов)
// ============================================================================

void ProcessAllFrame(
	uint8_t** matrix,
	uint8_t* d_fastRows, uint8_t* d_matrix,
	size_t wordsCount, size_t codeLength, size_t frameLength,
	size_t& matrixRows, size_t infoLength,
	int* cwIndexFast, int* rowType, int* pivotsMatrix,
	int* d_pivotsMatrix, int* d_foundRow, int* d_pivotOut)
{
	for (int rowIndex = 0; rowIndex < (int)infoLength; ++rowIndex) {
		bool found = false;

		while (!found) {
			// ================================================================
			// 1. ИЩЕМ ТОЧНОЕ СОВПАДЕНИЕ (pivot == rowIndex)
			// Сначала в RAM (rowType == 0), потом в Cache (rowType == 1)
			// ================================================================
			int selectedRowIndex = -1;

			// Проход по RAM
			for (size_t i = 0; i < wordsCount; ++i) {
				if (rowType[i] == 0 && cwIndexFast[i] == rowIndex) {
					selectedRowIndex = (int)i;
					found = true;
					break;
				}
			}

			// Проход по Cache, если в RAM не нашли
			if (!found) {
				for (size_t i = 0; i < wordsCount; ++i) {
					if (rowType[i] == 1 && cwIndexFast[i] == rowIndex) {
						selectedRowIndex = (int)i;
						found = true;
						break;
					}
				}
			}

			if (found) {
				// Добавляем строку в итоговую матрицу
				cudaMemcpy(d_matrix + matrixRows * frameLength,
					d_fastRows + selectedRowIndex * frameLength,
					frameLength, cudaMemcpyDeviceToDevice);

				pivotsMatrix[matrixRows] = rowIndex;
				cudaMemcpy(d_pivotsMatrix + matrixRows, &rowIndex, sizeof(int), cudaMemcpyHostToDevice);

				matrixRows++;
				rowType[selectedRowIndex] = -1; // Помечаем как отработанную (Dead)
				break;
			}

			// ================================================================
			// 2. ЕСЛИ ТОЧНОГО СОВПАДЕНИЯ НЕТ, ИЩЕМ СТРОКУ ДЛЯ РЕДУКЦИИ (pivot < rowIndex)
			// Сначала проверяем RAM, затем Cache.
			// ================================================================
			int candidateRow = -1;

			// Ищем в RAM строку, чей пивот уже «пройден» (< rowIndex)
			for (size_t i = 0; i < wordsCount; ++i) {
				if (rowType[i] == 0 && cwIndexFast[i] != -1 && cwIndexFast[i] < rowIndex) {
					candidateRow = (int)i;
					break;
				}
			}

			// Если в RAM нет, ищем в Cache
			if (candidateRow == -1) {
				for (size_t i = 0; i < wordsCount; ++i) {
					if (rowType[i] == 1 && cwIndexFast[i] != -1 && cwIndexFast[i] < rowIndex) {
						candidateRow = (int)i;
						break;
					}
				}
			}

			if (candidateRow != -1) {
				int currentPivot = cwIndexFast[candidateRow];

				// Ищем базовую строку в d_matrix с таким же пивотом
				int init = INT_MAX;
				cudaMemcpy(d_foundRow, &init, sizeof(int), cudaMemcpyHostToDevice);

				int blocks = (int)((matrixRows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
				if (blocks < 1) blocks = 1;

				FindPivotKernel << <blocks, THREADS_PER_BLOCK >> > (
					d_pivotsMatrix, (int)matrixRows, currentPivot, d_foundRow);

				int target_row;
				cudaMemcpy(&target_row, d_foundRow, sizeof(int), cudaMemcpyDeviceToHost);

				if (target_row != INT_MAX) {
					// Делаем XOR
					int blocksXor = (int)((frameLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
					if (blocksXor < 1) blocksXor = 1;

					XorRowsKernel << <blocksXor, THREADS_PER_BLOCK >> > (
						d_fastRows, d_matrix, (int)frameLength, candidateRow, target_row, (int)codeLength);

					// Пересчитываем пивот
					FindPivotInSingleRow << <1, THREADS_PER_BLOCK >> > (
						d_fastRows + candidateRow * frameLength, (int)frameLength, d_pivotOut);

					int newPivot;
					cudaMemcpy(&newPivot, d_pivotOut, sizeof(int), cudaMemcpyDeviceToHost);

					cwIndexFast[candidateRow] = newPivot;

					// Если пивот стал -1, отключаем строку
					if (newPivot == -1) {
						rowType[candidateRow] = -1;
					}
					// Если пивот стал больше rowIndex, переводим строку в разряд Cache
					else if (newPivot > rowIndex) {
						rowType[candidateRow] = 1;
					}

					continue; // Рекурсивно повторяем поиск для текущего rowIndex
				}
			}

			// ================================================================
			// 3. ЕСЛИ ПИВОТ ОКАЗАЛСЯ СЛИШКОМ СПРАВА (pivot > rowIndex)
			// Проверяем, есть ли в RAM свежие строки, которые опережают текущий rowIndex.
			// Если есть — переводим их в кэш, чтобы они дождались своего часа.
			// ================================================================
			bool movedToCache = false;
			for (size_t i = 0; i < wordsCount; ++i) {
				if (rowType[i] == 0 && cwIndexFast[i] > rowIndex) {
					rowType[i] = 1; // Переводим в кэш
					movedToCache = true;
				}
			}

			if (movedToCache) {
				continue; // Пробуем снова найти совпадение для этого rowIndex
			}

			// ================================================================
			// 4. НИЧЕГО НЕ ПОДОШЛО — ФОРМИРУЕМ НУЛЕВУЮ СТРОКУ
			// ================================================================
			memset(matrix[rowIndex], 0, frameLength);
			pivotsMatrix[rowIndex] = -1;
			matrixRows++;
			int m1 = -1;
			cudaMemcpy(d_pivotsMatrix + rowIndex, &m1, sizeof(int), cudaMemcpyHostToDevice);
			break;
		}
	}

	// Обратный ход Гаусса
	ReverseGearGPU(d_matrix, matrixRows, frameLength, d_pivotsMatrix, (int)codeLength);

	for (size_t i = 0; i < infoLength; ++i)
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

	// Параметры
	const string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\6.21.bin)";
	const string TempDir = R"(D:\Rubin\sessions\tmp_1783328386069\)";

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

	// Матрица результата
	uint8_t** matrix = new uint8_t*[infoLength];
	for (size_t i = 0; i < infoLength; ++i)
		matrix[i] = new uint8_t[frameLength]();

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

	// Индексы и типы строк
	int* cwIndexFast = new int[wordsCount];
	int* rowType = new int[wordsCount]; // 0 = RAM, 1 = Cache, -1 = Dead
	for (size_t i = 0; i < wordsCount; ++i) {
		cwIndexFast[i] = -1;
		rowType[i] = 0; // Изначально все строки относятся к RAM
	}

	int* pivotsMatrix = new int[infoLength];
	for (size_t i = 0; i < infoLength; ++i)
		pivotsMatrix[i] = -1;

	// GPU-память
	uint8_t *d_fastRows = nullptr, *d_matrix = nullptr;
	int *d_pivots = nullptr, *d_pivotsMatrix = nullptr;
	int *d_foundRow = nullptr, *d_pivotOut = nullptr;

	cudaMalloc(&d_fastRows, totalBytes);
	cudaMemcpy(d_fastRows, linearRows, totalBytes, cudaMemcpyHostToDevice);
	cudaFreeHost(linearRows);
	linearRows = nullptr;

	cudaMalloc(&d_matrix, infoLength * frameLength);
	cudaMemset(d_matrix, 0, infoLength * frameLength);

	cudaMalloc(&d_pivots, wordsCount * sizeof(int));

	// Первичный поиск ведущих единиц
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
	cudaMalloc(&d_pivotOut, sizeof(int));

	// Гаусс
	size_t matrixRows = 0;

	ProcessAllFrame(
		matrix,
		d_fastRows, d_matrix,
		wordsCount, codeLength, frameLength,
		matrixRows, infoLength,
		cwIndexFast, rowType, pivotsMatrix,
		d_pivotsMatrix, d_foundRow, d_pivotOut);

	// Запись результата
	string outName = TempDir + (dataType ? "result.bis" : "result.bin");
	WriteResultToFile(outName, matrix, matrixRows, codeLength, dataType != 0);

	// Очистка
	for (size_t i = 0; i < infoLength; ++i)
		delete[] matrix[i];
	delete[] matrix;
	delete[] cwIndexFast;
	delete[] rowType;
	delete[] pivotsMatrix;

	cudaFree(d_fastRows);
	cudaFree(d_matrix);
	cudaFree(d_pivotsMatrix);
	cudaFree(d_foundRow);
	cudaFree(d_pivotOut);

	auto ms = chrono::duration_cast<chrono::milliseconds>(
		chrono::high_resolution_clock::now() - start).count();
	cout << "Время выполнения программы в мс: " << ms << endl;

	return 0;
}