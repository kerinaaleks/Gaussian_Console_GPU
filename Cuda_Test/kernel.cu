#include <iostream>
#include <fstream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

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

__global__ void ReverseGearKernel(
	uint8_t* d_matrix,
	int matrixRows,
	int frameLength,
	int pivotRow,
	int* d_pivotsMatrix) {

	int j = blockIdx.x; // Индекс строки которую обрабатываем (индекс текущего блока)
	int tid = threadIdx.x; // Индекс байта который обрабатываем

	if (j >= matrixRows)
		return;

	if (j == pivotRow)
		return;

	uint8_t* rowJ = d_matrix + j * frameLength;
	const uint8_t* rowI = d_matrix + pivotRow * frameLength;

	int pivotColumn = d_pivotsMatrix[pivotRow];

	if (pivotColumn < 0) return;

	if (!GetBitDevice(rowJ, pivotColumn))
		return;

	if (tid < frameLength)
		((uint8_t*)rowJ)[tid] ^= rowI[tid];
}

__global__ void CheckZeroKernel(const uint8_t* rows, int frameLength, bool* result) {
	int rowIndex = blockIdx.x; //строка
	int tid = threadIdx.x; //байт внутри строки

	__shared__ bool isZero;

	if (tid == 0)
		isZero = true; // считаем строку нулевой
	__syncthreads();

	const uint8_t* row = rows + rowIndex * frameLength;

	if (tid < frameLength && row[tid] != 0) // если хоть один байт не равен 0, строка не нулевая
		isZero = false;
	__syncthreads();

	if (tid == 0) // поток 0 записывает результат
		result[rowIndex] = isZero;

}

int FindLeadingOne(uint8_t* frameBuffer, size_t codeLength) {

	for (size_t i = 0; i < codeLength; i++) {
		if (GetBitHost(frameBuffer, i)) {
			return i;// вернули индекс где встретили первую единицу в этой строке
		}
	}
	return -1;//вот этим  числом сообщим что строка пустая
}

void ReverseGearGPU(
	uint8_t* d_matrix,
	size_t matrixRows,
	size_t frameLength,
	int* d_pivotsMatrix)
{
	int threadPerBlock = 256;

	for (int i = (int)matrixRows - 1; i >= 0; --i) {
		int blocks = i;

		ReverseGearKernel << <blocks, threadPerBlock >> > (d_matrix, (int)matrixRows, (int)frameLength, i, d_pivotsMatrix);

		cudaDeviceSynchronize();
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

__global__ void FindPivotRowKernel(const int* pivotsMatrix, int matrixRows, int pivot, int*result) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;//вычисление глобального индекса потока

	if (tid >= matrixRows)// проверка выхода за границы
		return;

	if (pivotsMatrix[tid] == pivot)
		atomicMin(result, tid);
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
	int pivot,          // текущий pivot этой строки
	int* cwIndexFast,   // pivot'ы строк в RAM
	int* cwIndexCache,  // pivot'ы строк в cache
	int* pivotsMatrix,  // pivot'ы строк итоговой матрицы (CPU)
	int* d_pivotsMatrix,// pivot'ы строк итоговой матрицы (GPU)
	int* d_foundRow     // буфер для найденной строки на GPU
)
{
	if (pivot == -1)
		return false;

	while (true)
	{
		// 1. pivot == rowIndex → строка идёт в матрицу
		if (pivot == rowIndex) {
			memcpy(matrix[rowIndex], frameBuffer, frameLength);
			cudaMemcpy(d_matrix + rowIndex * frameLength,
				frameBuffer,
				frameLength,
				cudaMemcpyHostToDevice);
			pivotsMatrix[rowIndex] = pivot;
			matrixRows++;
			cudaMemcpy(d_pivotsMatrix + rowIndex,
				&pivotsMatrix[rowIndex],
				sizeof(int),
				cudaMemcpyHostToDevice);
			return true;
		}

		// 2. pivot < rowIndex → XOR с уже существующей строкой матрицы
		if (pivot < rowIndex) {
			// ищем строку с таким pivot на GPU
			int init = INT_MAX;
			int target_row;

			cudaMemcpy(d_foundRow, &init, sizeof(int), cudaMemcpyHostToDevice);

			int threads = 256;
			int blocks = (int)((matrixRows + threads - 1) / threads);

			FindPivotRowKernel << <blocks, threads >> > (d_pivotsMatrix,
				(int)matrixRows,
				pivot,
				d_foundRow);
			cudaDeviceSynchronize();

			cudaMemcpy(&target_row, d_foundRow, sizeof(int), cudaMemcpyDeviceToHost);

			if (target_row == INT_MAX)
				target_row = -1;

			if (target_row == -1) // нет строки с таким pivot в матрице
				return false;

			int threadsPerBlock = 256;
			int blocksXor = (int)((frameLength + threadsPerBlock - 1) / threadsPerBlock);

			XorRowsKernel << <blocksXor, threadsPerBlock >> > (
				d_fastRows,
				d_cache,
				d_matrix,
				(int)frameLength,
				srcIndex,
				fromCache,
				target_row);
			cudaDeviceSynchronize();

			cudaMemcpy(
				frameBuffer,
				(fromCache ? d_cache : d_fastRows) + srcIndex * frameLength,
				frameLength,
				cudaMemcpyDeviceToHost
			);

			pivot = FindLeadingOne(frameBuffer, codeLength);
			if (pivot == -1)
				return false;

			// проверяем эту же строку ещё раз, но уже изменённую
			continue;
		}

		// 3. pivot > rowIndex → кладём строку в кэш
		if (cacheRows < wordsCount) {
			cache[cacheRows] = new uint8_t[frameLength];
			memcpy(cache[cacheRows], frameBuffer, frameLength);
			cwIndexCache[cacheRows] = pivot;
			cacheRows++;
			cudaMemcpy(d_cache + (cacheRows - 1) * frameLength,
				cache[cacheRows - 1],
				frameLength,
				cudaMemcpyHostToDevice);
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
	int* pivotsMatrix,
	bool* zeroFlags,
	int* d_pivotsMatrix,
	int* d_foundRow)
{
	for (int rowIndex = 0; rowIndex < (int)infoLength; ++rowIndex) {
		cout << "итерация: " << rowIndex << endl;
		bool found = false;

		// Сначала ищем строку в RAM
		for (size_t i = 0; i < wordsCount && !found; ++i) {
			if (cwIndexFast[i] == -1)
				continue; // строки нет (пустая или уже использована)
			// проверяем, что строка не нулевая
			/*bool isZero = true;
			for (size_t b = 0; b < frameLength; ++b) {
				if (fastRows[i][b] != 0) {
					isZero = false;
					break;
				}
			}
			if (isZero) {
				cwIndexFast[i] = -1;
				continue;
			}*/

			if (zeroFlags[i]) {
				cwIndexFast[i] = -1;
				continue;
			}
			memcpy(frameBuffer, fastRows[i], frameLength);
			int pivot = cwIndexFast[i];

			if (ProcessFrameGPU(
				(int)i,
				false, d_fastRows, d_cache, d_matrix, frameBuffer, matrix, cache, codeLength, frameLength, matrixRows, cacheRows, rowIndex, wordsCount, pivot, cwIndexFast, cwIndexCache, pivotsMatrix, d_pivotsMatrix, d_foundRow))
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
						for (size_t k = j + 1; k < cacheRows; ++k) {
							cudaMemcpy(d_cache + (k - 1) * frameLength,
								d_cache + k * frameLength,
								frameLength,
								cudaMemcpyDeviceToDevice);
						}
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
					for (size_t k = j + 1; k < cacheRows; ++k) {
						cudaMemcpy(d_cache + (k - 1) * frameLength,
							d_cache + k * frameLength,
							frameLength,
							cudaMemcpyDeviceToDevice);
					}
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
					pivotsMatrix,
					d_pivotsMatrix, d_foundRow))
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

			cudaMemcpy(d_pivotsMatrix + rowIndex,
				&pivotsMatrix[rowIndex],
				sizeof(int),
				cudaMemcpyHostToDevice);
		}
	}

	ReverseGearGPU(d_matrix, matrixRows, frameLength, d_pivotsMatrix);

	for (size_t i = 0; i < infoLength; ++i) {
		cudaMemcpy(matrix[i], d_matrix + i * frameLength, frameLength, cudaMemcpyDeviceToHost);
	}
}
//////////////////////////////////////////////////////////////////////
void PrintMatrix(uint8_t** matrix, size_t matrixRows, size_t codeLength) {
	cout << "Матрица после метода Гаусса\n\n";
	for (size_t i = 0; i < matrixRows; i++) {
		for (size_t j = 0; j < codeLength; j++) {
			cout << GetBitHost(matrix[i], j) << ' ';
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

	string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\6.17.bin)";
	string TempDir = R"(D:\Rubin\sessions\tmp_1783328386069\)";
	size_t codeLength = 2064;
	size_t infoLength = 2000;

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

	// pivot-ы RAM и cache
	int* cwIndexFast = new int[wordsCount];
	int* cwIndexCache = new int[wordsCount];
	for (size_t i = 0; i < wordsCount; ++i) {
		cwIndexFast[i] = -1;
		cwIndexCache[i] = -1;
	}

	// Линейный массив строк для GPU
	size_t totalBytes = wordsCount * frameLength;
	uint8_t* linearRows = new uint8_t[totalBytes];
	for (size_t i = 0; i < wordsCount; i++)
		memcpy(linearRows + i * frameLength, fastRows[i], frameLength);

	// ---------------- GPU ALLOC ----------------

	// d_fastRows — строки RAM на GPU
	uint8_t* d_fastRows;
	cudaMalloc(&d_fastRows, totalBytes);
	cudaMemcpy(d_fastRows, linearRows, totalBytes, cudaMemcpyHostToDevice);

	// d_cache — строки cache на GPU
	uint8_t* d_cache;
	cudaMalloc(&d_cache, wordsCount * frameLength);
	cudaMemset(d_cache, 0, wordsCount * frameLength);

	// d_matrix — итоговая матрица на GPU
	uint8_t* d_matrix;
	cudaMalloc(&d_matrix, infoLength * frameLength);
	cudaMemset(d_matrix, 0, infoLength * frameLength);

	// d_pivots — pivots для всех строк RAM
	int* d_pivots;
	cudaMalloc(&d_pivots, wordsCount * sizeof(int));

	// pivotsMatrix — pivots итоговой матрицы (CPU)
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

	// копируем pivots обратно
	int* pivots = new int[wordsCount];
	cudaMemcpy(pivots, d_pivots, wordsCount * sizeof(int), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < wordsCount; ++i)
		cwIndexFast[i] = pivots[i];
	// ---------------- GPU: CheckZeroKernel ----------------
	bool* d_zeroFlags;
	cudaMalloc(&d_zeroFlags, wordsCount * sizeof(bool));
	int blocksZero = (int)wordsCount;

	CheckZeroKernel << <blocksZero, threadsPerBlock >> > (d_fastRows, frameLength, d_zeroFlags);
	cudaDeviceSynchronize();

	bool* zeroFlags = new bool[wordsCount];
	cudaMemcpy(zeroFlags, d_zeroFlags, wordsCount * sizeof(bool), cudaMemcpyDeviceToHost);
	// ---------------- GPU: FindPivotRowKernel ----------------
	int* d_pivotsMatrix;
	cudaMalloc(&d_pivotsMatrix, infoLength * sizeof(int));
	cudaMemcpy(d_pivotsMatrix, pivotsMatrix, infoLength * sizeof(int), cudaMemcpyHostToDevice);

	int *d_foundRow;
	cudaMalloc(&d_foundRow, sizeof(int));

	// ---------------- Гаусс ----------------

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
		pivotsMatrix,
		zeroFlags,
		d_pivotsMatrix,
		d_foundRow);

	WriteMatrixToFile(TempDir + "result.bin", matrix, matrixRows, frameLength);
	//PrintMatrix(matrix, matrixRows, codeLength);


	// ---------------- Очистка ----------------

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

	cudaFree(d_pivots);
	cudaFree(d_fastRows);
	cudaFree(d_cache);
	cudaFree(d_matrix);

	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
	cout << "Время выполнения программы в мс: " << duration.count() << endl;

	return 0;
}
