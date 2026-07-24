#include <iostream>
#include <cmath>
#include <ctime>
#include <utility>
#include <clocale>
#include <algorithm>
#include <fstream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;

__global__ void FindLeadingOneKernel(const uint8_t* __restrict__  rows, int* __restrict__ pivots, int frameLength, int wordsCount) {
	int laneId = threadIdx.x & 31; // номер потока внутри варпа
	int warpId = threadIdx.x >> 5; // номер варпа внутри блока //побитовый сдвиг вправо на 5(деление на 32)
	int warpsPerBlock = blockDim.x >> 5; //сколько варпов в блоке
	int rowIndex = blockIdx.x * warpsPerBlock + warpId; // глобальный индекс строки 

	if (rowIndex >= wordsCount)
		return;

	const uint8_t* row = rows + rowIndex * frameLength; // указатель на начало строки
	int localPivot = INT_MAX; //каждый поток проверяет свой кусок строки
	const int chankSize = 2;// шаг - сколько байтов провряет один поток
	int startByte = laneId * chankSize;
	int endByte = startByte + chankSize;

	if (endByte > frameLength)
		endByte = frameLength;

	for (int b = startByte; b < endByte; b++) {
		uint8_t val = row[b];// один байт строки
		if (val != 0) {
			for (int bit = 0; bit < 8; bit++) {
				if (val & (1 << bit)) { //с права на лево
					localPivot = b * 8 + bit;
					break;
				}
			}

			break;
		}
	}
	// каждый поток нашел пивот в своем участке строки, warp‑reduce - находит мин пивот всех потоков,
	for (int offset = 16; offset > 0; offset >>= 1) {// offset >>= 1 -  деление на 2(сдвиг на 1) (классический warp‑reduce шаг)
		int other = __shfl_down_sync(0xFFFFFFFF, localPivot, offset); // поток читает значение из другого потока внутри того же варпа (Дай мне значение переменной localPivot из потока, который находится ниже меня на offset позиций)
		localPivot = min(localPivot, other);// по сути сравнивам пары
	}

	if (laneId == 0) {// записываем результат
		if (localPivot == INT_MAX)
			pivots[rowIndex] = -1;
		else
			pivots[rowIndex] = localPivot;
	}
}

__global__ void XorRowsKernel(
	uint8_t* d_fastRows,
	uint8_t* d_cache,
	uint8_t* d_matrix,
	int frameLength,
	int srcIndex, // Индекс строки, которую нужно изменить
	bool fromCache,
	int targetRow // Индекс строки матрицы с которой XOR‑им
) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x; // нормер байта строки, который должен обрботать этот поток

	if (tid >= frameLength) // Проверка выхода за границы
		return;

	uint8_t* src = fromCache ? (d_cache + srcIndex * frameLength) : (d_fastRows + srcIndex * frameLength); // строка пришла из кэша или из рама?
	uint8_t* target = d_matrix + targetRow * frameLength; // Выбор строки матрицы

	src[tid] ^= target[tid];
}

bool GetBit(const uint8_t* data, size_t bitIndex) {
	size_t byte = bitIndex / 8;// у какого байта
	size_t bit = bitIndex % 8;// какой бит берем
	return (data[byte] >> bit) & 1; // возвращаем бит
}

int FindLeadingOne(uint8_t* frameBuffer, size_t codeLength) {

	for (size_t i = 0; i < codeLength; i++) {
		if (GetBit(frameBuffer, i)) {
			return i;// вернули индекс где встретили первую единицу в этой строке
		}
	}
	return -1;//вот этим  числом сообщим что строка пустая
}

void ReverseGear(uint8_t** matrix, size_t matrixRows, size_t codeLength, size_t frameLength)
{
	for (int i = (int)matrixRows - 1; i > 0; i--) {
		for (int j = 0; j < i; j++) {
			if (GetBit(matrix[j], i)) {
				for (size_t k = 0; k < frameLength; k++)
					matrix[j][k] ^= matrix[i][k];
			}
		}
	}
}

void ReadCodeWords(ifstream& file, size_t cwIndex, size_t codeLength, uint8_t* dst) {
	// Номер бита во всем файле с которого начинается это слово
	size_t bitStart = cwIndex * codeLength; // номер кодового слова * длина одного кодового слова в битах
	size_t byteStart = bitStart / 8; // Номер байта с которого нужно начать чтение
	size_t bitOffset = bitStart % 8; // Сколько бит уже занято внитри этого байта предыдущими словами
	size_t bytesToRead = (bitOffset + codeLength + 7) / 8; // считаем сколько байт нужно что бы покрыть начальное смещение и все битвы слова

	// нужен для того, что бы мы в дальнейшем вытаскивали биты из байтов
	uint8_t* buf = new uint8_t[bytesToRead]; // Выделяем временный буфер

	file.seekg(byteStart);// ставим курсор
	file.read((char*)buf, bytesToRead);// читаем

	size_t frameLength = (codeLength + 7) / 8;
	// dst - выходной массив, куда полодим биты в упакованном виде(8бит в байте)
	memset(dst, 0, frameLength);// обнуляем

	// Побитовое извление и упаковка
	for (size_t i = 0; i < codeLength; i++) {
		size_t srcByte = (bitOffset + i) / 8; //Номер байта
		size_t srcBit = 7 - ((bitOffset + i) % 8); // номер бита внутри байта

		bool bit = (buf[srcByte] >> srcBit) & 1;// Сдвигаем байт так что бы нужный бит оказался в младше разряде
		if (bit) {
			size_t dstByte = i / 8;// номер байта в выходном массиве
			size_t dstBit = 7 - (i % 8); // номер бита внутри этого байта
			dst[dstByte] |= (1 << dstBit); // устанавливаем этот бит взн-ие 1
		}
	}
	delete[] buf;
}

bool ProcessFrameGPU(
	int srcIndex,
	bool fromCache,
	uint8_t* d_fastRows,
	uint8_t* d_cache,
	uint8_t* d_matrix,
	uint8_t* frameBuffer,
	uint8_t** matrix,
	uint8_t** cache,
	size_t codeLength,
	size_t frameLength,
	size_t& matrixRows,
	size_t& cacheRows,
	int rowIndex,
	size_t wordsCount,
	int pivot, // текущий pivot этой строки
	int* cwIndexFast, // pivot'ы строк в RAM
	int* cwIndexCache, // pivot'ы строк в cache
	int* pivotsMatrix // pivot'ы строк итоговой матрицы
)
{
	if (pivot == -1)
		return false;

	while (true)
	{
		// 1. pivot == rowIndex → строка идёт в матрицу
		if (pivot == rowIndex) {
			memcpy(matrix[rowIndex], frameBuffer, frameLength);
			cudaMemcpy(d_matrix + rowIndex * frameLength, frameBuffer, frameLength, cudaMemcpyHostToDevice);
			pivotsMatrix[rowIndex] = pivot;
			matrixRows++;
			return true;
		}

		// 2. pivot < rowIndex → XOR
		if (pivot < rowIndex) {
			int target_row = -1;
			for (size_t r = 0; r < matrixRows; ++r) {
				if (pivotsMatrix[r] == pivot) {
					target_row = (int)r;
					break;
				}
			}

			if (target_row == -1) // нет строки с таким pivot в матрице
				return false;

			int threadsPerBlock = 256;
			int blocks = (int)((frameLength + threadsPerBlock - 1) / threadsPerBlock);

			XorRowsKernel << <blocks, threadsPerBlock >> > (
				d_fastRows, d_cache, d_matrix, (int)frameLength, srcIndex, fromCache, target_row);
			cudaDeviceSynchronize();

			cudaMemcpy(
				frameBuffer,
				(fromCache ? d_cache : d_fastRows) + srcIndex * frameLength,
				frameLength, cudaMemcpyDeviceToHost
			);

			pivot = FindLeadingOne(frameBuffer, codeLength);
			if (pivot == -1)
				return false;

			continue;//проверяем снова эту же строку, но уже измененный ее вид
		}

		// 3. pivot > rowIndex → кладём строку в кэш
		if (cacheRows < wordsCount) {
			cache[cacheRows] = new uint8_t[frameLength];
			memcpy(cache[cacheRows], frameBuffer, frameLength);
			cwIndexCache[cacheRows] = pivot;
			cacheRows++;
		}

		return false;
	}
}

void ProcessAllFrame(
	uint8_t** matrix,
	uint8_t** cache,
	uint8_t** fastRows,
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
	int* pivotsMatrix)
{
	for (int rowIndex = 0; rowIndex < (int)infoLength; ++rowIndex) {
		cout << "итерация: " << rowIndex << endl;
		bool found = false;

		// Сначала ищем строку в RAM
		for (size_t i = 0; i < wordsCount && !found; ++i) {
			if (cwIndexFast[i] == -1)
				continue; // строки нет (пустая или уже использована)
			// проверяем, что строка не нулевая
			bool isZero = true;
			for (size_t b = 0; b < frameLength; ++b) {
				if (fastRows[i][b] != 0) {
					isZero = false;
					break;
				}
			}
			if (isZero) {
				cwIndexFast[i] = -1;
				continue;
			}

			memcpy(frameBuffer, fastRows[i], frameLength);
			int pivot = cwIndexFast[i];

			if (ProcessFrameGPU(
				(int)i,
				false, d_fastRows, d_cache, d_matrix, frameBuffer, matrix, cache, codeLength, frameLength, matrixRows, cacheRows, rowIndex, wordsCount, pivot, cwIndexFast, cwIndexCache, pivotsMatrix))
			{
				found = true;
				// строка RAM ушла в матрицу → очищаем её
				memset(fastRows[i], 0, frameLength);
				cwIndexFast[i] = -1;

				// пробуем заменить её строкой из кэша
				for (size_t j = 0; j < cacheRows; ++j) {
					if (cache[j] == nullptr)
						continue;

					bool cacheZero = true;
					for (size_t b = 0; b < frameLength; ++b) {
						if (cache[j][b] != 0) {
							cacheZero = false;
							break;
						}
					}
					if (cacheZero) {
						delete[] cache[j];
						cache[j] = nullptr;
						cwIndexCache[j] = -1;
						continue;
					}

					memcpy(fastRows[i], cache[j], frameLength);
					cwIndexFast[i] = cwIndexCache[j];

					delete[] cache[j];
					// сдвиг кэша и индексов
					for (size_t k = j + 1; k < cacheRows; ++k) {
						cache[k - 1] = cache[k];
						cwIndexCache[k - 1] = cwIndexCache[k];
					}

					cacheRows--;
					cache[cacheRows] = nullptr;
					cwIndexCache[cacheRows] = -1;

					break;
				}
			}
		}

		// Если RAM не дала строку → ищем в кэше
		if (!found) {
			for (size_t i = 0; i < cacheRows; ++i) {
				if (cache[i] == nullptr || cwIndexCache[i] == -1)
					continue;

				memcpy(frameBuffer, cache[i], frameLength);
				int pivot = cwIndexCache[i];

				if (ProcessFrameGPU(
					(int)i,
					true,
					d_fastRows,
					d_cache,
					d_matrix,
					frameBuffer,
					matrix,
					cache,
					codeLength,
					frameLength,
					matrixRows,
					cacheRows,
					rowIndex,
					wordsCount,
					pivot,
					cwIndexFast,
					cwIndexCache,
					pivotsMatrix))
				{
					delete[] cache[i];
					// сдвигаем кэш и индексы
					for (size_t j = i + 1; j < cacheRows; ++j) {
						cache[j - 1] = cache[j];
						cwIndexCache[j - 1] = cwIndexCache[j];
					}

					cacheRows--;
					cache[cacheRows] = nullptr;
					cwIndexCache[cacheRows] = -1;

					found = true;
					break;
				}
			}
		}

		// Если ни RAM, ни кэш не дали строку → нулевая строка
		if (!found) {
			memset(matrix[rowIndex], 0, frameLength);
			pivotsMatrix[rowIndex] = -1;
			matrixRows++;
		}
	}

	ReverseGear(matrix, infoLength, codeLength, frameLength);
}
//////////////////////////////////////////////////////////////////////
void PrintMatrix(uint8_t** matrix, size_t matrixRows, size_t codeLength) {
	cout << "Матрица после метода Гаусса\n\n";
	for (size_t i = 0; i < matrixRows; i++) {
		for (size_t j = 0; j < codeLength; j++) {
			cout << static_cast<int>(GetBit(matrix[i], j)) << ' ';
		}
		cout << '\n';
	}
}

void WriteMatrixToFile(const string& fileName, uint8_t** matrix, size_t matrixRows, size_t frameLength)
{
	ofstream out(fileName, ios::binary);

	if (!out.is_open()) {
		cout << "Ошибка открытия файла\n";
		return;
	}
	for (size_t i = 0; i < matrixRows; i++) {
		out.write(reinterpret_cast<char*>(matrix[i]), frameLength);
	}
	out.close();
}
////////////////////////////////////////////////////////////////////

int main()
{
	setlocale(LC_ALL, "");

	auto start = chrono::high_resolution_clock::now();

	string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\11.16.bin)";
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

	// CPU-матрица
	uint8_t** matrix = new uint8_t*[infoLength];
	for (size_t i = 0; i < infoLength; i++) {
		matrix[i] = new uint8_t[frameLength];
		memset(matrix[i], 0, frameLength);
	}

	// CPU-кэш
	uint8_t** cache = new uint8_t*[wordsCount];
	for (size_t i = 0; i < wordsCount; i++)
		cache[i] = nullptr;

	// CPU-RAM (fastRows)
	uint8_t** fastRows = new uint8_t*[wordsCount];
	for (size_t i = 0; i < wordsCount; i++) {
		fastRows[i] = new uint8_t[frameLength];
		ReadCodeWords(file, i, codeLength, fastRows[i]);
	}

	// pivot-ы
	int* cwIndexFast = new int[wordsCount];
	int* cwIndexCache = new int[wordsCount];
	for (size_t i = 0; i < wordsCount; ++i) {
		cwIndexFast[i] = -1;
		cwIndexCache[i] = -1;
	}

	// линейный массив строк
	size_t totalBytes = wordsCount * frameLength;
	uint8_t* linearRows = new uint8_t[totalBytes];
	for (size_t i = 0; i < wordsCount; i++)
		memcpy(linearRows + i * frameLength, fastRows[i], frameLength);

	// GPU: d_rows для pivot-ов
	uint8_t* d_rows;
	cudaMalloc(&d_rows, totalBytes);
	cudaMemcpy(d_rows, linearRows, totalBytes, cudaMemcpyHostToDevice);

	int* d_pivots;
	cudaMalloc(&d_pivots, wordsCount * sizeof(int));

	int* pivotsMatrix = new int[infoLength];
	for (size_t i = 0; i < infoLength; ++i)
		pivotsMatrix[i] = -1;

	int threadsPerBlock = 256;
	int warpsPerBlock = threadsPerBlock / 32;
	int blocks = (int)((wordsCount + warpsPerBlock - 1) / warpsPerBlock);

	FindLeadingOneKernel << <blocks, threadsPerBlock >> > (
		d_rows, d_pivots, (int)frameLength, (int)wordsCount);
	cudaDeviceSynchronize();

	int* pivots = new int[wordsCount];
	cudaMemcpy(pivots, d_pivots, wordsCount * sizeof(int), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < wordsCount; ++i)
		cwIndexFast[i] = pivots[i];

	// GPU: d_fastRows / d_cache / d_matrix
	uint8_t* d_fastRows;
	cudaMalloc(&d_fastRows, totalBytes);
	cudaMemcpy(d_fastRows, linearRows, totalBytes, cudaMemcpyHostToDevice);

	uint8_t* d_cache;
	cudaMalloc(&d_cache, wordsCount * frameLength);
	cudaMemset(d_cache, 0, wordsCount * frameLength);

	uint8_t* d_matrix;
	cudaMalloc(&d_matrix, infoLength * frameLength);
	cudaMemset(d_matrix, 0, infoLength * frameLength);

	// Гаусс
	ProcessAllFrame(
		matrix,
		cache,
		fastRows,
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
		pivotsMatrix);

	WriteMatrixToFile(TempDir + "result.bin", matrix, matrixRows, frameLength);
	//PrintMatrix(matrix, matrixRows, codeLength);

	// очистка
	file.close();

	for (size_t i = 0; i < infoLength; i++)
		delete[] matrix[i];
	delete[] matrix;

	for (size_t i = 0; i < wordsCount; i++)
		delete[] fastRows[i];
	delete[] fastRows;

	for (size_t i = 0; i < cacheRows; i++)
		if (cache[i] != nullptr)
			delete[] cache[i];
	delete[] cache;

	delete[] frameBuffer;
	delete[] cwIndexFast;
	delete[] cwIndexCache;
	delete[] pivotsMatrix;
	delete[] pivots;
	delete[] linearRows;

	cudaFree(d_rows);
	cudaFree(d_pivots);
	cudaFree(d_fastRows);
	cudaFree(d_cache);
	cudaFree(d_matrix);

	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
	cout << "Время выполнения программы в мс: " << duration.count() << endl;
	return 0;
}