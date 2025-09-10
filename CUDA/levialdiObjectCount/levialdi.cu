#include <csetjmp>
#include <curand_kernel.h>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <vector>
#define DIM 64
#define THREADS_PER_BLOCK 256

void displayPicture(const std::vector<bool> &pixels, int width, int height);

// Inicializacion de estados del generador de números aleatorios (cuRAND) para cada hilo...
__global__ void initCurandStates(curandState *states, unsigned long seed,int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    curand_init(seed, idx, 0, &states[idx]);
}

// Aleatorizar imagen...
__global__ void randomizeKernel(bool *pixels, float density,curandState *states, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    curandState localState = states[idx];
    pixels[idx] = (curand_uniform(&localState) < density);
    states[idx] = localState;
  }
}

// Pasada de crecimiento
// Evita race conditions con d_in y d_out
__global__ void growthKernel(const bool *d_in, bool *d_out, int width,int height, float growthChance,curandState *states) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n = width * height;

  if (idx >= n)
    return;

  if (d_in[idx]) {
    d_out[idx] = true;
    return;
  }

  int x = idx % width;
  int y = idx / width;
  int neighbourCount = 0;

  for (int ny = -1; ny <= 1; ++ny) {
    for (int nx = -1; nx <= 1; ++nx) {

      if (nx == 0 && ny == 0)
        continue;

      int checkX = x + nx;
      int checkY = y + ny;

      if (checkX >= 0 && checkX < width && checkY >= 0 && checkY < height)
        if (d_in[checkY * width + checkX])
          neighbourCount++;
    }
  }

  if (neighbourCount > 0) {
    curandState localState = states[idx];
    float totalGrowthChance = neighbourCount * growthChance;

    if (curand_uniform(&localState) < totalGrowthChance) {
      d_out[idx] = true;
    } else {
      d_out[idx] = false;
    }
    states[idx] = localState;
  } else {
    d_out[idx] = false;
  }
}

// Algoritmo de Levialdi
__global__ void levialdiKernel(const bool *d_in, bool *d_out, int width,int height, int *d_objectCount, int *d_changed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n = width * height;
  if (idx >= n)
    return;

  // Coordenadas para manejar los bordes
  int x = idx % width;
  int y = idx / width;

  // Definición de los 8 vecinos
  bool P = d_in[idx];
  bool N = (y > 0) ? d_in[idx - width] : false;
  bool NE = (y > 0 && x + 1 < width) ? d_in[idx - width + 1] : false;
  bool E = (x + 1 < width) ? d_in[idx + 1] : false;
  bool SE = (x + 1 < width && y + 1 < height) ? d_in[idx + 1 + width] : false;
  bool S = (y + 1 < height) ? d_in[idx + width] : false;
  bool SW = (y + 1 < height && x > 0) ? d_in[idx + width - 1] : false;
  bool W = (x > 0) ? d_in[idx - 1] : false;
  bool NW = (x > 0 && y > 0) ? d_in[idx - width - 1] : false;

  bool all_neighbours_zero = !(N || NE || E || SE || S || SW || W || NW);

  // Condición de conteo
  if (P && all_neighbours_zero) {
    atomicAdd(d_objectCount, 1);
  }

  // Condición de reducción
  bool term1 = (int)N + (int)P + (int)E > 1;
  bool term2 = (int)P + (int)NE > 1;
  bool new_P = (term1 || term2);
  d_out[idx] = new_P;

  // Verificar modificaciones para identificar finalización del algoritmo
  if (P != new_P)
    atomicExch(d_changed, 1);
}

class CudaBinaryPicture {
private:
  int width;
  int height;
  int numPixels;

  bool *d_pixels;
  curandState *d_randStates;

public:
  // Constructor
  CudaBinaryPicture(int w, int h) : width(w), height(h) {
    numPixels = w * h;

    cudaMalloc(&d_pixels, numPixels * sizeof(bool));
    cudaMalloc(&d_randStates, numPixels * sizeof(curandState));

    int blocks = (numPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    initCurandStates<<<blocks, THREADS_PER_BLOCK>>>(d_randStates, time(0),numPixels);
  }

  // Destructor
  ~CudaBinaryPicture() {
    cudaFree(d_pixels);
    cudaFree(d_randStates);
  }

  void setPixel(int x, int y, bool value) {
    if (x < 0 || x >= width || y < 0 || y >= height)
      return;
    int idx = x + y * width;
    cudaMemcpy(d_pixels + idx, &value, sizeof(bool), cudaMemcpyHostToDevice);
  }

  bool getPixel(int x, int y) {
    if (x < 0 || x >= width || y < 0 || y >= height)
      return false;

    int idx = x + y * width;
    bool h_pixel_value;
    cudaMemcpy(&h_pixel_value, d_pixels + idx, sizeof(bool),cudaMemcpyDeviceToHost);
    return h_pixel_value;
  }

  void randomize(float density) {
    int blocks = (numPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    randomizeKernel<<<blocks, THREADS_PER_BLOCK>>>(d_pixels, density,d_randStates, numPixels);
    cudaDeviceSynchronize();
  }

  void randomizeAndGrow(float seedDensity, int growthPasses,float growthChance) {
    this->randomize(seedDensity);
    bool *d_in, *d_out;
    int blocks = (numPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc(&d_in, numPixels * sizeof(bool));
    d_out = d_pixels;

    for (int pass = 0; pass < growthPasses; ++pass) {
      std::swap(d_in, d_out);
      growthKernel<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_out, width, height,growthChance, d_randStates);
      cudaDeviceSynchronize();
    }

    if (growthPasses % 2 != 0)
      cudaMemcpy(d_pixels, d_in, numPixels * sizeof(bool),cudaMemcpyDeviceToDevice);

    cudaFree(d_in);
  }

  int countObjectsLevialdi() {
    bool *d_in, *d_out;
    int *d_objectCount, *d_changed;

    cudaMalloc(&d_in, numPixels * sizeof(bool));
    cudaMalloc(&d_objectCount, sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    d_out = d_pixels;

    cudaMemset(d_objectCount, 0, sizeof(int));

    int blocks = (numPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int h_changed = 1;
    int iteration = 1;

    while (h_changed) {
      cudaMemset(d_changed, 0, sizeof(int));
      levialdiKernel<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_out, width, height,d_objectCount, d_changed);
      cudaDeviceSynchronize();
      cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

      // impimir la imagen después de cada iteración del algoritmo de Levialdi
      std::cout << "=======================================";
      std::cout << "=======================================" << std::endl;
      std::vector<bool> h_mid_image = this->getHostData();
      displayPicture(h_mid_image, width, height);
      // fin del código para imprimir la imagen después de cada iteración del algoritmo de Levialdi

      std::swap(d_in, d_out);
      iteration++;
    }

    int finalObjectCount;
    cudaMemcpy(&finalObjectCount, d_objectCount, sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(d_pixels, d_out, numPixels * sizeof(bool),cudaMemcpyDeviceToDevice);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_objectCount);
    cudaFree(d_changed);

    return finalObjectCount;
  }

  std::vector<bool> getHostData() const {
    std::vector<char> h_temp_buffer(numPixels);
    cudaMemcpy(h_temp_buffer.data(), d_pixels, numPixels * sizeof(bool),cudaMemcpyDeviceToHost);
    std::vector<bool> h_pixels(numPixels);
    for (int i = 0; i < numPixels; ++i)
      h_pixels[i] = (h_temp_buffer[i] != 0);
    return h_pixels;
  }
};

void displayPicture(const std::vector<bool> &pixels, int width, int height) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x)
      std::cout << (pixels[x + y * width] ? "██" : "  ");
    std::cout << std::endl;
  }
}

int main() {
  CudaBinaryPicture picture(DIM, DIM);

  picture.randomizeAndGrow(0.005f, 4, 0.15f);

  std::vector<bool> initialPicture = picture.getHostData();
  displayPicture(initialPicture, DIM, DIM);

  int objectCount = picture.countObjectsLevialdi();

  std::cout << "\n Número de objetos encontrados: " << objectCount << std::endl;

  return 0;
}
