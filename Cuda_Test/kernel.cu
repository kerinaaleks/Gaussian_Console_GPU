#include <iostream>
#include <cmath>
#include <ctime>
#include <utility>
#include <clocale>
#include <algorithm>
#include <fstream>

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

int ComputePivotGPU(uint8_t* frameBuffer, int frameLength) {
	uint8_t* d_row;
	cudaMalloc(&d_row, frameLength);
	cudaMemcpy(d_row, frameBuffer, frameLength, cudaMemcpyHostToDevice);

	int*d_pivot;
	cudaMalloc(&d_pivot, sizeof(int));

	FindLeadingOneKernel << <1, 32 >> > (d_row, d_pivot, frameLength, 1);

	int pivot;
	cudaMemcpy(&pivot, d_pivot, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_row);
	cudaFree(d_pivot);

	return pivot;
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

///////////////////////////////////////////////////////ВОТ ЕЕ ДОПИСАЛА В ПТ
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
////////////////////////////////////////////////////////////////////////////////////////
bool ProcessFrame(
	uint8_t* frameBuffer,
	uint8_t** matrix,
	uint8_t** cache,
	size_t codeLength,
	size_t frameLength,
	size_t& matrixRows,
	size_t& cacheRows,
	int rowIndex,
	size_t infoLength,
	size_t wordsCount,
	int pivot,           // текущий pivot этой строки
	int* cwIndexFast,     // pivot'ы строк в RAM
	int* cwIndexCache,    // pivot'ы строк в cache
	int* pivotsMatrix     // pivot'ы строк итоговой матрицы
)
{
	if (pivot == -1)
		return false;

	// 1. pivot == rowIndex → строка идёт в матрицу
	if (pivot == rowIndex) {
		memcpy(matrix[rowIndex], frameBuffer, frameLength);
		pivotsMatrix[rowIndex] = pivot;
		matrixRows++;
		return true;
	}

	// 2. pivot < rowIndex → пытаемся привести XOR по строке с таким же pivot
	if (pivot < rowIndex) {
		int target_row = -1;
		for (size_t r = 0; r < matrixRows; ++r) {
			if (pivotsMatrix[r] == pivot) {
				target_row = (int)r;
				break;
			}
		}

		if (target_row == -1)
			return false; // нет строки с таким pivot в матрице

		// XOR с найденной строкой
		for (size_t j = 0; j < frameLength; ++j)
			frameBuffer[j] ^= matrix[target_row][j];

		// пересчитываем pivot этой строки (CPU)
		//pivot = FindLeadingOne(frameBuffer, codeLength);
		pivot = ComputePivotGPU(frameBuffer, codeLength);
		if (pivot == -1)
			return false;

		// рекурсивно пробуем ещё раз с новым pivot
		return ProcessFrame(frameBuffer, matrix, cache,
			codeLength, frameLength,
			matrixRows, cacheRows,
			rowIndex, infoLength, wordsCount,
			pivot, cwIndexFast, cwIndexCache, pivotsMatrix);
	}

	// 3. pivot > rowIndex → кладём строку в кэш
	if (cacheRows < wordsCount) {
		cache[cacheRows] = new uint8_t[frameLength];
		memcpy(cache[cacheRows], frameBuffer, frameLength);
		cwIndexCache[cacheRows] = pivot;   // запоминаем pivot строки в кэше
		cacheRows++;
	}

	return false;
}

void ProcessAllFrame(
	uint8_t** matrix,
	uint8_t** cache,
	uint8_t** fastRows,
	uint8_t* frameBuffer,
	size_t wordsCount,
	size_t codeLength,
	size_t frameLength,
	size_t& matrixRows,
	size_t& cacheRows,
	size_t infoLength,
	int* cwIndexFast,    // pivot'ы строк в RAM
	int* cwIndexCache,   // pivot'ы строк в cache
	int* pivotsMatrix    // pivot'ы строк итоговой матрицы
)
{
	for (int rowIndex = 0; rowIndex < (int)infoLength; ++rowIndex) {
		cout << "итерация: " << rowIndex << endl;
		bool found = false;

		// 1. Сначала ищем строку в RAM
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

			if (ProcessFrame(frameBuffer, matrix, cache, codeLength, frameLength, matrixRows, cacheRows, rowIndex, infoLength, wordsCount, pivot, cwIndexFast, cwIndexCache, pivotsMatrix))
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

					// перенос строки из cache[j] в fastRows[i]
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

		// 2. Если RAM не дала строку → ищем в кэше
		if (!found) {
			for (size_t i = 0; i < cacheRows; ++i) {
				if (cache[i] == nullptr || cwIndexCache[i] == -1)
					continue;

				memcpy(frameBuffer, cache[i], frameLength);
				int pivot = cwIndexCache[i];

				if (ProcessFrame(frameBuffer, matrix, cache,
					codeLength, frameLength,
					matrixRows, cacheRows,
					rowIndex, infoLength, wordsCount,
					pivot, cwIndexFast, cwIndexCache, pivotsMatrix))
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

		// 3. Если ни RAM, ни кэш не дали строку → нулевая строка
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

	string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\6.17.bin)";
	string Type = "bin";
	size_t FileBitLength = 940928; // Беззнаковый целочисленный типа(8 байт для 64 битных систем)
	size_t FirstBit = 0;
	string TempDir = R"(D:\Rubin\sessions\tmp_1783328386069\)"; //папка куда запишем выходной файл
	size_t codeLength = 2064;//длина кодового слова
	size_t infoLength = 2000;//длина инфомрационной части КС

	// массив для конечной матрицы со смещением
	size_t frameLength = (codeLength + 7) / 8; // Вычисляем сколько нужно выделить байт под наши биты из которых состоит слово. // Аналог size_t frameLength = ceil(codeLength/8.0);
	//size_t frameLength = ceil(codeLength / 8.0);
	uint8_t *frameBuffer = new uint8_t[frameLength];//конечный вид строчки хранится тут
	//size_t wordsCount = FileBitLength / codeLength;//число строк( сколько слов в сообщении)

	size_t matrixRows = 0;//сколько строк уже записано в матрицу
	size_t cacheRows = 0; // сколько строк сейчас находится в кэше
	size_t leadElement = 0;//текущий ожидаемый ведущий столбец

	ifstream file(InputFileName, ios::binary);
	if (!file.is_open()) {
		cout << "Ошибка открытия файла" << endl;
		return false;
	}
	else
		cout << "Файл открыт" << endl;

	file.seekg(0, ios::end);
	size_t fileSizeBytes = file.tellg();
	file.seekg(0, ios::beg);

	size_t fileSizeBits = fileSizeBytes * 8;
	size_t wordsCount = fileSizeBits / codeLength;

	uint8_t **matrix = new uint8_t*[infoLength];//массив для гауса
	for (int i = 0; i < infoLength; i++) {
		matrix[i] = new uint8_t[frameLength];// выделяем байты
		memset(matrix[i], 0, frameLength);
	}

	uint8_t **cache = new uint8_t*[wordsCount];// массив для кэша
	for (int i = 0; i < wordsCount; i++) {
		cache[i] = nullptr;// пока не будем выделять фулом память, просто заполним нулями
	}

	// Считали все входное сообщение, и разделилил его на слова и записали в рам
	uint8_t** fastRows = new uint8_t *[wordsCount];
	for (int i = 0; i < wordsCount; i++) {
		fastRows[i] = new uint8_t[frameLength];
		ReadCodeWords(file, i, codeLength, fastRows[i]);
	}

	////////////
	int* cwIndexFast = new int[wordsCount];
	int* cwIndexCache = new int[wordsCount];

	for (int i = 0; i < wordsCount; ++i) {
		cwIndexFast[i] = -1;     // исходный индекс слова
		cwIndexCache[i] = -1;    // пока пусто
	}
	// Создадим линейный массив строк
	size_t totalBytes = wordsCount * frameLength; // Общее кол-во байтов всех строк
	uint8_t* linearRows = new uint8_t[totalBytes]; // Один большой линейный массив, куда мы упаковываем все строки подряд

	for (size_t i = 0; i < wordsCount; i++) {
		memcpy(linearRows + i * frameLength, fastRows[i], frameLength);//копируем строки
	}

	// Выделим память на GPU
	uint8_t *d_rows;
	cudaMalloc(&d_rows, totalBytes);
	// Скопируем строки на GPU
	cudaMemcpy(d_rows, linearRows, totalBytes, cudaMemcpyHostToDevice);
	// Выделим массив pivot‑ов на GPU
	int* d_pivots;
	cudaMalloc(&d_pivots, wordsCount * sizeof(int)); // sizeof(int) - возвращает размер целочисленного типа данных в байтах

	int* pivotsMatrix = new int[infoLength];
	for (size_t i = 0; i < infoLength; ++i)
		pivotsMatrix[i] = -1;

	//Вычисляем блоки и потоки
	int threadsPerBlock = 256;
	int warpsPerBlock = threadsPerBlock / 32;
	//int blocks = ceil(wordsCount / warpsPerBlock); //может не сработать, можно попробовать int blocks = (wordsCount + warpsPerBlock - 1) / warpsPerBlock;
	int blocks = (wordsCount + warpsPerBlock - 1) / warpsPerBlock;

	FindLeadingOneKernel << <blocks, threadsPerBlock >> > (d_rows, d_pivots, frameLength, wordsCount);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		cout << "CUDA ERROR: " << cudaGetErrorString(err) << endl;
	cout << "Посчитали первые единицы " << endl;

	int* pivots = new int[wordsCount]; // индекс первой единицы строки fastRows
	cudaMemcpy(pivots, d_pivots, wordsCount * sizeof(int), cudaMemcpyDeviceToHost);
	
	for (size_t i = 0; i < wordsCount; ++i) {
		cwIndexFast[i] = pivots[i];
	}

	//cout << "Массив первых единиц (pivot'ы):\n";
	//for (size_t i = 0; i < wordsCount; ++i) {
	//	cout << "строка " << i << ": " << pivots[i] << '\n';
	//}


	ProcessAllFrame(matrix, cache, fastRows, frameBuffer, wordsCount, codeLength, frameLength, matrixRows, cacheRows, infoLength, cwIndexFast, cwIndexCache, pivotsMatrix);
	WriteMatrixToFile(TempDir + "result.bin", matrix, matrixRows, frameLength); 
	PrintMatrix(matrix, matrixRows, codeLength);
	

	//////////////////////////////////////////////
	file.close();
	for (size_t i = 0; i < infoLength; i++)
		delete[] matrix[i];
	delete[] matrix;

	for (size_t i = 0; i < cacheRows; i++) {
		if (cache[i] != nullptr)
			delete[] cache[i];
	}
	delete[] cache;
	/////////////////////////////////////////////
}
