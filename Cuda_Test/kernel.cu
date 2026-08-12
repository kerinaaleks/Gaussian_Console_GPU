#include <fstream>
#include <chrono>
#include <cstring>
#include <climits>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iomanip>
#include <vector>
using namespace std;
// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ УСТРОЙСТВЕННЫЕ ФУНКЦИИ
// ============================================================================
__device__ __forceinline__ uint8_t GetTailMask(int frameLength, int codeLength)
{
	int excess = frameLength * 8 - codeLength;
	if (excess > 0 && excess < 8)
		return (uint8_t)((1u << (8 - excess)) - 1u);
	return 0xFF;
}
__device__ __forceinline__ bool GetBitDevice(const uint8_t* row, int bitIndex)
{
	return (row[bitIndex / 8] >> (bitIndex % 8)) & 1;
}
inline bool GetBitHost(const uint8_t* row, size_t bitIndex)
{
	return (row[bitIndex / 8] >> (bitIndex % 8)) & 1;
}
// ============================================================================
// ЯДРА
// ============================================================================
__global__ void MaskLastByteKernel(uint8_t* d_data, int frameLength, uint8_t mask, int rowCount)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowIdx >= rowCount) return;
	d_data[rowIdx * frameLength + (frameLength - 1)] &= mask;
}
__global__ void FindLeadingOneKernel(const uint8_t* __restrict__ rows,
	int* __restrict__ pivots,
	int frameLength,
	int wordsCount,
	int codeLength)
{
	int laneId = threadIdx.x & 31;
	int warpId = threadIdx.x >> 5;
	int warpsPerBlock = blockDim.x >> 5;
	int rowIndex = blockIdx.x * warpsPerBlock + warpId;
	if (rowIndex >= wordsCount) return;
	const uint8_t* row = rows + rowIndex * frameLength;
	int localPivot = INT_MAX;
	uint8_t mask = GetTailMask(frameLength, codeLength);
	int validBits = codeLength % 8;
	if (validBits == 0) validBits = 8;
	for (int b = laneId; b < frameLength; b += 32)
	{
		uint8_t val = row[b];
		if (b == frameLength - 1)
			val &= mask;
		if (val != 0)
		{
			int bitsToCheck = (b == frameLength - 1) ? validBits : 8;
			for (int bit = 0; bit < bitsToCheck; ++bit)
			{
				if (val & (1 << bit))
				{
					localPivot = min(localPivot, b * 8 + bit);
					break;
				}
			}
		}
	}
	for (int offset = 16; offset > 0; offset >>= 1)
	{
		int other = __shfl_down_sync(0xFFFFFFFFu, localPivot, offset);
		localPivot = min(localPivot, other);
	}
	if (laneId == 0)
		pivots[rowIndex] = (localPivot == INT_MAX) ? -1 : localPivot;
}
__global__ void FindPivotInSingleRow(const uint8_t* row,
	int frameLength,
	int* pivotOut,
	int codeLength,
	int startBit)
{
	int tid = threadIdx.x;
	int laneId = tid & 31;
	int warpId = tid >> 5;
	__shared__ int warpMin[8];
	int localPivot = INT_MAX;
	uint8_t mask = GetTailMask(frameLength, codeLength);
	int validBits = codeLength % 8;
	if (validBits == 0) validBits = 8;
	int startByte = startBit / 8;
	int startInByte = startBit % 8;
	for (int b = tid; b < frameLength; b += blockDim.x)
	{
		if (b < startByte) continue;
		uint8_t v = row[b];
		if (b == startByte)
			v &= (uint8_t)(0xFFu << startInByte);
		if (b == frameLength - 1)
			v &= mask;
		if (v != 0)
		{
			int bitsToCheck = (b == frameLength - 1) ? validBits : 8;
			for (int bit = 0; bit < bitsToCheck; ++bit)
			{
				if (v & (1 << bit))
				{
					localPivot = min(localPivot, b * 8 + bit);
					break;
				}
			}
		}
	}
	for (int off = 16; off > 0; off >>= 1)
		localPivot = min(localPivot, __shfl_down_sync(0xFFFFFFFFu, localPivot, off));
	if (laneId == 0)
		warpMin[warpId] = localPivot;
	__syncthreads();
	if (tid == 0)
	{
		int fin = INT_MAX;
		for (int w = 0; w < (blockDim.x >> 5); ++w)
			fin = min(fin, warpMin[w]);
		*pivotOut = (fin == INT_MAX) ? -1 : fin;
	}
}
__global__ void XorRowsKernel(uint8_t* d_fastRows,
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
	uint8_t* src = fromCache ? (d_cache + srcIndex * frameLength)
		: (d_fastRows + srcIndex * frameLength);
	uint8_t* target = d_matrix + targetRow * frameLength;
	src[tid] ^= target[tid];
	// КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: после XOR всегда чистим хвост
	if (tid == frameLength - 1)
	{
		uint8_t mask = GetTailMask(frameLength, codeLength);
		src[tid] &= mask;
	}
}
__global__ void ReverseGearKernel(
	uint8_t* d_matrix,
	int frameLength,
	int pivotRow,
	int codeLength)
{
	int j = blockIdx.x;
	if (j >= pivotRow) return;
	uint8_t* rowJ = d_matrix + (size_t)j * frameLength;
	uint8_t* rowI = d_matrix + (size_t)pivotRow * frameLength;
	if (!GetBitDevice(rowJ, pivotRow))
		return;
	// XOR
	for (int idx = threadIdx.x; idx < frameLength; idx += blockDim.x)
		rowJ[idx] ^= rowI[idx];
	// ОБЯЗАТЕЛЬНО ждём, пока ВСЕ потоки закончат XOR
	__syncthreads();
	// Только после барьера чистим хвост
	if (threadIdx.x == 0)
	{
		int excess = frameLength * 8 - codeLength;
		if (excess > 0 && excess < 8)
		{
			uint8_t mask = (uint8_t)((1u << (8 - excess)) - 1u);
			rowJ[frameLength - 1] &= mask;
		}
	}
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
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = start; i < count - 1; ++i)
			arr[i] = arr[i + 1];
	}
}
// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ CPU
// ============================================================================
void ReverseGearGPU(uint8_t* d_matrix,
	size_t matrixRows,
	size_t frameLength,
	int* d_pivotsMatrix,
	int codeLength,
	uint8_t** matrix,
	size_t infoLength)
{
	int threadPerBlock = 256;
	for (int i = (int)infoLength - 1; i > 0; --i)
	{
		ReverseGearKernel << <i, threadPerBlock >> > (d_matrix, (int)frameLength, i, codeLength);
	}
	cudaDeviceSynchronize();
	// Финальная очистка хвостов (на всякий случай)
	int excessBits = (int)frameLength * 8 - codeLength;
	if (excessBits > 0 && excessBits < 8)
	{
		uint8_t mask = (uint8_t)((1u << (8 - excessBits)) - 1u);
		int threads = 256;
		int blocks = ((int)infoLength + threads - 1) / threads;
		MaskLastByteKernel << <blocks, threads >> > (d_matrix, (int)frameLength, mask, (int)infoLength);
		cudaDeviceSynchronize();
	}
	// Внутри ReverseGearGPU, после MaskLastByteKernel и перед циклом cudaMemcpy:
	// Жёсткая очистка на хосте тоже (на всякий случай)
	for (size_t r = 0; r < infoLength; ++r)
	{
		cudaMemcpy(matrix[r], d_matrix + r * frameLength, frameLength, cudaMemcpyDeviceToHost);
		int excess = (int)frameLength * 8 - codeLength;
		if (excess > 0 && excess < 8)
			matrix[r][frameLength - 1] &= (uint8_t)((1u << (8 - excess)) - 1u);
	}
	for (size_t r = 0; r < infoLength; ++r)
		cudaMemcpy(matrix[r], d_matrix + r * frameLength, frameLength, cudaMemcpyDeviceToHost);
}
void ReadCodeWords(ifstream& file, size_t cwIndex, size_t codeLength, uint8_t* dst)
{
	size_t bitStart = cwIndex * codeLength;
	size_t byteStart = bitStart / 8;
	size_t bitOffset = bitStart % 8;
	size_t bytesToRead = (bitOffset + codeLength + 7) / 8;
	uint8_t* buf = new uint8_t[bytesToRead];
	file.seekg(byteStart);
	file.read((char*)buf, bytesToRead);
	size_t frameLength = (codeLength + 7) / 8;
	memset(dst, 0, frameLength);
	if (bitOffset > 0)
	{
		for (size_t i = 0; i < codeLength; ++i)
		{
			size_t srcByte = (bitOffset + i) / 8;
			size_t srcBit = (bitOffset + i) % 8;
			bool bit = (buf[srcByte] >> srcBit) & 1;
			if (bit)
			{
				size_t dstByte = i / 8;
				size_t dstBit = i % 8;
				dst[dstByte] |= (1 << dstBit);
			}
		}
	}
	else
	{
		memcpy(dst, buf, frameLength);
		// Явное маскирование хвоста (как в эталонном GetFrameFromBinFile)
		int excessBits = (int)frameLength * 8 - (int)codeLength;
		switch (excessBits)
		{
		case 1: dst[frameLength - 1] &= 0x7F; break;
		case 2: dst[frameLength - 1] &= 0x3F; break;
		case 3: dst[frameLength - 1] &= 0x1F; break;
		case 4: dst[frameLength - 1] &= 0x0F; break;
		case 5: dst[frameLength - 1] &= 0x07; break;
		case 6: dst[frameLength - 1] &= 0x03; break;
		case 7: dst[frameLength - 1] &= 0x01; break;
		default: break;
		}
	}
	// Гарантируем, что хвост чистый даже после побитового копирования
	
		int excessBits = (int)frameLength * 8 - (int)codeLength;
		if (excessBits > 0 && excessBits < 8)
			dst[frameLength - 1] &= (uint8_t)((1u << (8 - excessBits)) - 1u);
	
	delete[] buf;
}
// ============================================================================
// ОСНОВНАЯ ЛОГИКА
// ============================================================================
bool ProcessFrameGPU(int srcIndex,
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
	int* d_cwIndexCache)
{
	if (pivot == -1)
	{
		if (fromCache)
		{
			cwIndexCache[srcIndex] = -1;
			int m1 = -1;
			cudaMemcpy(d_cwIndexCache + srcIndex, &m1, sizeof(int), cudaMemcpyHostToDevice);
		}
		else
			cwIndexFast[srcIndex] = -1;
		return false;
	}
	uint8_t* srcPtr = fromCache ? (d_cache + (size_t)srcIndex * frameLength)
		: (d_fastRows + (size_t)srcIndex * frameLength);
	int startBit = 0;
	FindPivotInSingleRow << <1, 256 >> > (srcPtr, (int)frameLength, d_pivotOut, (int)codeLength, startBit);
	cudaDeviceSynchronize();
	cudaMemcpy(&pivot, d_pivotOut, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(frameBuffer, srcPtr, frameLength, cudaMemcpyDeviceToHost);
	// Гарантируем чистый хвост на хосте
	{
		int excess = (int)frameLength * 8 - (int)codeLength;
		if (excess > 0 && excess < 8)
			frameBuffer[frameLength - 1] &= (uint8_t)((1u << (8 - excess)) - 1u);
	}
	if (fromCache)
	{
		cwIndexCache[srcIndex] = pivot;
		cudaMemcpy(d_cwIndexCache + srcIndex, &pivot, sizeof(int), cudaMemcpyHostToDevice);
	}
	else
		cwIndexFast[srcIndex] = pivot;
	if (pivot == -1)
	{
		if (fromCache)
		{
			cwIndexCache[srcIndex] = -1;
			int m1 = -1;
			cudaMemcpy(d_cwIndexCache + srcIndex, &m1, sizeof(int), cudaMemcpyHostToDevice);
		}
		else
			cwIndexFast[srcIndex] = -1;
		return false;
	}
	while (true)
	{
		if (pivot == rowIndex)
		{
			// Ещё раз чистим хвост перед записью в матрицу
			{
				int excess = (int)frameLength * 8 - (int)codeLength;
				if (excess > 0 && excess < 8)
					frameBuffer[frameLength - 1] &= (uint8_t)((1u << (8 - excess)) - 1u);
			}
			memcpy(matrix[rowIndex], frameBuffer, frameLength);
			cudaMemcpy(d_matrix + (size_t)rowIndex * frameLength, frameBuffer, frameLength, cudaMemcpyHostToDevice);
			pivotsMatrix[rowIndex] = pivot;
			matrixRows++;
			cudaMemcpy(d_pivotsMatrix + rowIndex, &pivot, sizeof(int), cudaMemcpyHostToDevice);
			return true;
		}
		if (pivot < rowIndex)
		{
			int targetRow = pivot;
			if (pivotsMatrix[targetRow] >= 0)
			{
				int threadsPerBlock = 256;
				int blocksXor = (int)((frameLength + threadsPerBlock - 1) / threadsPerBlock);
				XorRowsKernel << <blocksXor, threadsPerBlock >> > (
					d_fastRows, d_cache, d_matrix,
					(int)frameLength, srcIndex, fromCache, targetRow, (int)codeLength);
				cudaDeviceSynchronize();
			}
			startBit = pivot + 1;
			FindPivotInSingleRow << <1, 256 >> > (srcPtr, (int)frameLength, d_pivotOut, (int)codeLength, startBit);
			cudaDeviceSynchronize();
			cudaMemcpy(&pivot, d_pivotOut, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(frameBuffer, srcPtr, frameLength, cudaMemcpyDeviceToHost);
			// Чистим хвост после копирования
			{
				int excess = (int)frameLength * 8 - (int)codeLength;
				if (excess > 0 && excess < 8)
					frameBuffer[frameLength - 1] &= (uint8_t)((1u << (8 - excess)) - 1u);
			}
			if (fromCache)
			{
				cwIndexCache[srcIndex] = pivot;
				cudaMemcpy(d_cwIndexCache + srcIndex, &pivot, sizeof(int), cudaMemcpyHostToDevice);
			}
			else
				cwIndexFast[srcIndex] = pivot;
			if (pivot == -1)
			{
				if (fromCache)
				{
					cwIndexCache[srcIndex] = -1;
					int m1 = -1;
					cudaMemcpy(d_cwIndexCache + srcIndex, &m1, sizeof(int), cudaMemcpyHostToDevice);
				}
				else
					cwIndexFast[srcIndex] = -1;
				return false;
			}
			continue;
		}
		if (pivot > rowIndex)
		{
			if (fromCache)
			{
				cwIndexCache[srcIndex] = pivot;
				cudaMemcpy(d_cwIndexCache + srcIndex, &pivot, sizeof(int), cudaMemcpyHostToDevice);
				return false;
			}
			else
			{
				if (cacheRows < wordsCount)
				{
					// Чистим хвост перед записью в кэш
					{
						int excess = (int)frameLength * 8 - (int)codeLength;
						if (excess > 0 && excess < 8)
							frameBuffer[frameLength - 1] &= (uint8_t)((1u << (8 - excess)) - 1u);
					}
					cudaMemcpy(d_cache + cacheRows * frameLength, frameBuffer, frameLength, cudaMemcpyHostToDevice);
					cwIndexCache[cacheRows] = pivot;
					cudaMemcpy(d_cwIndexCache + cacheRows, &pivot, sizeof(int), cudaMemcpyHostToDevice);
					cacheRows++;
					cwIndexFast[srcIndex] = -1;
				}
				return false;
			}
		}
	}
}
bool ProcessFrameHostExact(
	uint8_t* row, // текущая строка (изменяется)
	int zeroCount, // нужный столбец = rowIndex
	uint8_t** matrix, // уже построенные строки матрицы
	size_t frameLength,
	size_t codeLength)
{
	int startPos = 0;
	while (true)
	{
		int findedPos = -1;
		// Ищем первую единицу начиная с startPos
		for (int pos = startPos; pos < (int)codeLength; ++pos)
		{
			if (GetBitHost(row, pos))
			{
				findedPos = pos;
				break;
			}
		}
		if (findedPos < 0)
			return false; // строка полностью нулевая
		if (findedPos == zeroCount)
			return true; // строка подходит
		if (findedPos < zeroCount)
		{
			// XOR с уже готовой строкой матрицы
			for (size_t b = 0; b < frameLength; ++b)
				row[b] ^= matrix[findedPos][b];
			// Обязательно чистим хвост
			int excess = (int)frameLength * 8 - (int)codeLength;
			if (excess > 0 && excess < 8)
				row[frameLength - 1] &= (uint8_t)((1u << (8 - excess)) - 1u);
			//startPos = findedPos + 1;
			continue;
		}
		// findedPos > zeroCount — пока не подходит
		return false;
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
	int* d_pivotOut)
{
	// Сначала загружаем все строки из d_fastRows на хост (для удобства)
	// (можно оставить и на устройстве, но для точного совпадения проще на хосте)
	uint8_t** hostFastRows = new uint8_t*[wordsCount];
	for (size_t i = 0; i < wordsCount; ++i)
	{
		hostFastRows[i] = new uint8_t[frameLength];
		cudaMemcpy(hostFastRows[i], d_fastRows + i * frameLength, frameLength, cudaMemcpyDeviceToHost);
		// Чистим хвост сразу
		int excess = (int)frameLength * 8 - (int)codeLength;
		if (excess > 0 && excess < 8)
			hostFastRows[i][frameLength - 1] &= (uint8_t)((1u << (8 - excess)) - 1u);
	}
	// Кэш тоже будем держать на хосте для простоты
	std::vector<uint8_t*> hostCache;
	std::vector<int> hostCachePivot; // можно не использовать
	for (int rowIndex = 0; rowIndex < (int)infoLength; ++rowIndex)
	{
		bool found = false;
		// 1. Ищем подходящую строку среди fastRows
		for (size_t i = 0; i < wordsCount && !found; ++i)
		{
			if (cwIndexFast[i] == -1) continue; // уже использована
			// Копируем строку во временный буфер
			memcpy(frameBuffer, hostFastRows[i], frameLength);
			if (ProcessFrameHostExact(frameBuffer, rowIndex, matrix, frameLength, codeLength))
			{
				// Строка подошла
				memcpy(matrix[rowIndex], frameBuffer, frameLength);
				cudaMemcpy(d_matrix + (size_t)rowIndex * frameLength, frameBuffer, frameLength, cudaMemcpyHostToDevice);
				pivotsMatrix[rowIndex] = rowIndex;
				matrixRows++;
				cwIndexFast[i] = -1; // помечаем как использованную
				found = true;
			}
			else
			{
				// Строка не подошла — можно отправить в кэш
				// (упрощённо пока просто оставляем)
			}
		}
		// 2. Если не нашли в fastRows — ищем в кэше
		if (!found)
		{
			for (size_t j = 0; j < hostCache.size() && !found; ++j)
			{
				memcpy(frameBuffer, hostCache[j], frameLength);
				if (ProcessFrameHostExact(frameBuffer, rowIndex, matrix, frameLength, codeLength))
				{
					memcpy(matrix[rowIndex], frameBuffer, frameLength);
					cudaMemcpy(d_matrix + (size_t)rowIndex * frameLength, frameBuffer, frameLength, cudaMemcpyHostToDevice);
					pivotsMatrix[rowIndex] = rowIndex;
					matrixRows++;
					// Удаляем из кэша
					delete[] hostCache[j];
					hostCache.erase(hostCache.begin() + j);
					found = true;
				}
			}
		}
		// 3. Если всё ещё не нашли — нулевая строка
		if (!found)
		{
			memset(matrix[rowIndex], 0, frameLength);
			cudaMemset(d_matrix + rowIndex * frameLength, 0, frameLength);
			pivotsMatrix[rowIndex] = -1;
			matrixRows++;
		}
		// Прогресс (опционально)
		if (rowIndex % 100 == 0)
			cout << "Прямой ход: " << rowIndex << " / " << infoLength << "\r" << flush;
	}
	cout << "\nПрямой ход завершён. matrixRows = " << matrixRows << endl;
	// Диагностика пивотов
	cout << "\n=== Диагностика прямого хода ===" << endl;
	int mismatch = 0;
	for (size_t r = 0; r < infoLength; ++r)
	{
		if (pivotsMatrix[r] != -1 && pivotsMatrix[r] != (int)r)
		{
			cout << "!!! row " << r << " pivot = " << pivotsMatrix[r] << endl;
			mismatch++;
		}
	}
	if (mismatch == 0)
		cout << "Пивоты в порядке." << endl;
	else
		cout << "Найдено смещений: " << mismatch << endl;
	// Освобождаем временные массивы
	for (size_t i = 0; i < wordsCount; ++i)
		delete[] hostFastRows[i];
	delete[] hostFastRows;
	for (auto p : hostCache)
		delete[] p;
	// Обратный ход пока отключаем (для чистого сравнения прямого хода)
	// ReverseGearGPU(...);
}
// ============================================================================
// ВВОД/ВЫВОД
// ============================================================================
void WritePackedBitsToFile(
	const string& fileName,
	uint8_t** matrix,
	size_t infoLength,
	size_t codeLength)
{
	ofstream out(fileName, ios::binary);
	if (!out.is_open()) return;
	size_t frameLength = (codeLength + 7) / 8;
	int excessBits = (int)(frameLength * 8 - codeLength);
	uint8_t tailMask = (excessBits > 0 && excessBits < 8)
		? (uint8_t)((1u << (8 - excessBits)) - 1u)
		: 0xFF;
	uint8_t* outBuf = new uint8_t[frameLength + 2];
	int restLen = 0;
	uint8_t restVal = 0;
	for (size_t row = 0; row < infoLength; ++row)
	{
		uint8_t* data = matrix[row];
		// Чистим хвост
		if (excessBits > 0)
			data[frameLength - 1] &= tailMask;
		size_t totalBitsNow = restLen + codeLength;
		size_t writeBytes = totalBitsNow / 8;
		int newRestLen = (int)(totalBitsNow % 8);
		memset(outBuf, 0, writeBytes + 1);
		int bitPos = 0;
		// 1. Старый остаток
		for (int i = 0; i < restLen; ++i)
		{
			if ((restVal >> i) & 1)
				outBuf[bitPos / 8] |= (uint8_t)(1u << (bitPos % 8));
			bitPos++;
		}
		// 2. Биты текущей строки
		for (size_t i = 0; i < codeLength; ++i)
		{
			if (GetBitHost(data, i))
				outBuf[bitPos / 8] |= (uint8_t)(1u << (bitPos % 8));
			bitPos++;
		}
		// Пишем полные байты
		out.write(reinterpret_cast<const char*>(outBuf), writeBytes);
		// Новый остаток
		restLen = newRestLen;
		restVal = (restLen > 0) ? outBuf[writeBytes] : 0;
	}
	// Дописываем последний остаток (как WriteRest)
	if (restLen > 0)
	{
		uint8_t lastByte = 0;
		for (int i = 0; i < restLen; ++i)
			if ((restVal >> i) & 1)
				lastByte |= (uint8_t)(1u << i);
		out.write(reinterpret_cast<const char*>(&lastByte), 1);
	}
	delete[] outBuf;
	out.close();
}
// ============================================================================
// MAIN
// ============================================================================
int main()
{
	setlocale(LC_ALL, "");
	auto start = chrono::high_resolution_clock::now();
	string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\6.21.bin)";
	string TempDir = R"(D:\Rubin\sessions\tmp_1783328386069)";
	size_t codeLength = 2004;
	size_t infoLength = 2000;
	size_t frameLength = (codeLength + 7) / 8;
	uint8_t* frameBuffer = new uint8_t[frameLength];
	size_t matrixRows = 0, cacheRows = 0;
	ifstream file(InputFileName, ios::binary);
	if (!file.is_open())
	{
		cerr << "Не удалось открыть входной файл\n";
		return 1;
	}
	file.seekg(0, ios::end);
	size_t fileSizeBytes = file.tellg();
	file.seekg(0, ios::beg);
	size_t wordsCount = (fileSizeBytes * 8) / codeLength;
	uint8_t** matrix = new uint8_t*[infoLength];
	for (size_t i = 0; i < infoLength; ++i)
	{
		matrix[i] = new uint8_t[frameLength];
		memset(matrix[i], 0, frameLength);
	}
	uint8_t** fastRows = new uint8_t*[wordsCount];
	for (size_t i = 0; i < wordsCount; ++i)
	{
		fastRows[i] = new uint8_t[frameLength];
		ReadCodeWords(file, i, codeLength, fastRows[i]);
	}
	size_t totalBytes = wordsCount * frameLength;
	uint8_t* linearRows = new uint8_t[totalBytes];
	for (size_t i = 0; i < wordsCount; ++i)
		memcpy(linearRows + i * frameLength, fastRows[i], frameLength);
	int* cwIndexFast = new int[wordsCount];
	int* cwIndexCache = new int[wordsCount];
	for (size_t i = 0; i < wordsCount; ++i)
	{
		cwIndexFast[i] = -1;
		cwIndexCache[i] = -1;
	}
	uint8_t *d_fastRows, *d_cache, *d_matrix;
	cudaMalloc(&d_fastRows, totalBytes);
	cudaMemcpy(d_fastRows, linearRows, totalBytes, cudaMemcpyHostToDevice);
	cudaMalloc(&d_cache, wordsCount * frameLength);
	cudaMemset(d_cache, 0, wordsCount * frameLength);
	cudaMalloc(&d_matrix, infoLength * frameLength);
	cudaMemset(d_matrix, 0, infoLength * frameLength);
	int *d_pivots;
	int *pivotsMatrix = new int[infoLength];
	int *d_pivotsMatrix, *d_foundRow, *d_cwIndexCache, *d_pivotOut;
	for (size_t i = 0; i < infoLength; ++i)
		pivotsMatrix[i] = -1;
	cudaMalloc(&d_pivots, wordsCount * sizeof(int));
	int threadsPerBlock = 256;
	int warpsPerBlock = threadsPerBlock / 32;
	int blocks = (int)((wordsCount + warpsPerBlock - 1) / warpsPerBlock);
	FindLeadingOneKernel << <blocks, threadsPerBlock >> > (
		d_fastRows, d_pivots, (int)frameLength, (int)wordsCount, (int)codeLength);
	cudaDeviceSynchronize();
	int* pivots = new int[wordsCount];
	cudaMemcpy(pivots, d_pivots, wordsCount * sizeof(int), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < wordsCount; ++i)
		cwIndexFast[i] = pivots[i];
	cudaMalloc(&d_pivotsMatrix, infoLength * sizeof(int));
	cudaMemcpy(d_pivotsMatrix, pivotsMatrix, infoLength * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&d_foundRow, sizeof(int));
	cudaMalloc(&d_cwIndexCache, wordsCount * sizeof(int));
	cudaMemcpy(d_cwIndexCache, cwIndexCache, wordsCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&d_pivotOut, sizeof(int));
	ProcessAllFrame(matrix, frameBuffer, d_fastRows, d_cache, d_matrix, wordsCount, codeLength, frameLength, matrixRows, cacheRows, infoLength, cwIndexFast, cwIndexCache, pivotsMatrix, d_pivotsMatrix, d_foundRow, d_cwIndexCache, d_pivotOut);
	//PrintMatrix(matrix, infoLength, codeLength);
	WritePackedBitsToFile(TempDir + "result.bin", matrix, infoLength, codeLength);
	cout << "\n=== Последние 16 бит строк (для сравнения с эталоном) ===" << endl;
	for (size_t r = 0; r < 30; ++r) // первые 30 строк
	{
		cout << "row " << setw(5) << r << ": ";
		for (int b = 15; b >= 0; --b)
		{
			size_t bit = codeLength - 1 - b;
			cout << (GetBitHost(matrix[r], bit) ? '1' : '0');
		}
		cout << endl;
	}
	// ===== ДИАГНОСТИКА ХВОСТА =====
	int excess = (int)frameLength * 8 - (int)codeLength;
	uint8_t mask = (excess > 0 && excess < 8) ? (uint8_t)((1u << (8 - excess)) - 1u) : 0xFF;
	cout << "\n=== Проверка последних байт матрицы ===" << endl;
	cout << "codeLength = " << codeLength << ", frameLength = " << frameLength
		<< ", excess = " << excess << ", mask = 0x" << hex << (int)mask << dec << endl;
	int badRows = 0;
	for (size_t r = 0; r < min(infoLength, (size_t)20); ++r) // первые 20 строк
	{
		uint8_t last = matrix[r][frameLength - 1];
		uint8_t high = last & ~mask; // то, что должно быть 0
		if (high != 0)
		{
			cout << "row " << r << " lastByte = 0x" << hex << (int)last
				<< " high bits = 0x" << (int)high << dec << endl;
			badRows++;
		}
	}
	if (badRows == 0)
		cout << "В первых 20 строках хвост чистый." << endl;
	else
		cout << "Найдено строк с мусором в хвосте: " << badRows << endl;
	// Очистка
	file.close();
	for (size_t i = 0; i < infoLength; ++i) delete[] matrix[i];
	delete[] matrix;
	for (size_t i = 0; i < wordsCount; ++i) delete[] fastRows[i];
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
	cout << "Время выполнения программы в мс: "
		<< chrono::duration_cast<chrono::milliseconds>(
			chrono::high_resolution_clock::now() - start).count()
		<< endl;
	return 0;
}