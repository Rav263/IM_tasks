#include <stdio.h>
// Функция сложения двух векторов
__global__ void addVector(float* left, float* right, float* result)
{
  //Получаем id текущей нити.
  int idx = threadIdx.x;
  
  //Расчитываем результат.
  result[idx] = left[idx] + right[idx];
}


#define SIZE 512
__host__ int main()
{
  //Выделяем память под вектора
  float* vec1 = new float[SIZE];
  float* vec2 = new float[SIZE];
  float* vec3 = new float[SIZE];

  //Инициализируем значения векторов
  for (int i = 0; i < SIZE; i++)
  {
    vec1[i] = i;
    vec2[i] = i;
  }

  //Указатели на память видеокарте
  float* devVec1;
  float* devVec2;
  float* devVec3;

  //Выделяем память для векторов на видеокарте
  cudaMalloc((void**)&devVec1, sizeof(float) * SIZE);
  cudaMalloc((void**)&devVec2, sizeof(float) * SIZE);
  cudaMalloc((void**)&devVec3, sizeof(float) * SIZE);

  //Копируем данные в память видеокарты
  cudaMemcpy(devVec1, vec1, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(devVec2, vec2, sizeof(float) * SIZE, cudaMemcpyHostToDevice);


  dim3 gridSize = dim3(1, 1, 1);    //Размер используемого грида
  dim3 blockSize = dim3(SIZE, 1, 1); //Размер используемого блока

  //Выполняем вызов функции ядра
  addVector<<<gridSize, blockSize>>>(devVec1, devVec2, devVec3);
    
    //Выполняем вызов функции ядра
  //addVector<<<blocks, threads>>>(devVec1, devVec2, devVec3);

  //Хендл event'а
  cudaEvent_t syncEvent;

  cudaEventCreate(&syncEvent);    //Создаем event
  cudaEventRecord(syncEvent, 0);  //Записываем event
  cudaEventSynchronize(syncEvent);  //Синхронизируем event

  //Только теперь получаем результат расчета
  cudaMemcpy(vec3, devVec3, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

   //Результаты расчета
  for (int i = 0; i < SIZE; i++)
  {
    printf("Element #%i: %.1f\n", i , vec3[i]);
  }

  //
  // Высвобождаем ресурсы
  //

  cudaEventDestroy(syncEvent);

  cudaFree(devVec1);
  cudaFree(devVec2);
  cudaFree(devVec3);

  delete[] vec1; vec1 = 0;
  delete[] vec2; vec2 = 0;
  delete[] vec3; vec3 = 0;
}

