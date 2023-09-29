#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

const double epsilon = 1e-9;


__global__ void addParallelGPU(double* A, double* B, double* C, int vectorSize, double alpha, double beta) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < vectorSize)
        C[idx] = alpha*A[idx] + beta*B[idx];
}


void addSequentCPU(double* A, double* B, double* C, int vectorSize, double alpha, double beta) {
    for (int i = 0; i < vectorSize; i++)
        C[i] = alpha*A[i] + beta*B[i];
}


int main(int argc, char *argv[]) {
    float sequent_duration, parallel_duration_pure, parallel_duration;
    cudaEvent_t sequent_start, sequent_stop, parallel_start_pure, parallel_stop_pure, parallel_start, parallel_stop;

    if (argc != 4) {
        cout << "Wrong arguments number: " << argc - 1 << endl;
        return 1;
    }

    const double alpha = stod(argv[1]), beta = stod(argv[2]);
    const int vectorSize = stoi(argv[3]);

    const size_t vectorBytesSize = vectorSize * sizeof(double);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

    cout << "Alpha: " << alpha << endl;
    cout << "Beta: " << beta << endl;
    cout << "Vector size: " << vectorSize << endl << endl;

    double *h_A = new double[vectorSize];
    double *h_B = new double[vectorSize];
    double *h_C = new double[vectorSize];
    double *ref_C = new double[vectorSize];

    for (int i = 0; i < vectorSize; i++) {
        h_A[i] = (double)rand() / RAND_MAX;
        h_B[i] = (double)rand() / RAND_MAX;
    }

    // ===== CPU sequent =====
    cudaEventCreate(&sequent_start);
    cudaEventCreate(&sequent_stop);
    cudaEventRecord(sequent_start);

    addSequentCPU(h_A, h_B, ref_C, vectorSize, alpha, beta);

    cudaEventRecord(sequent_stop);
    cudaEventSynchronize(sequent_stop);
    cudaEventElapsedTime(&sequent_duration, sequent_start, sequent_stop);
    // =======================

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, vectorBytesSize);
    cudaMalloc((void**)&d_B, vectorBytesSize);
    cudaMalloc((void**)&d_C, vectorBytesSize);

    // ===== GPU parallel =====
    cudaEventCreate(&parallel_start);
    cudaEventCreate(&parallel_stop);
    cudaEventRecord(parallel_start);

    cudaMemcpy(d_A, h_A, vectorBytesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, vectorBytesSize, cudaMemcpyHostToDevice);

    cudaEventCreate(&parallel_start_pure);
    cudaEventCreate(&parallel_stop_pure);
    cudaEventRecord(parallel_start_pure);

    addParallelGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, vectorSize, alpha, beta);

    cudaEventRecord(parallel_stop_pure);
    cudaEventSynchronize(parallel_stop_pure);
    cudaEventElapsedTime(&parallel_duration_pure, parallel_start_pure, parallel_stop_pure);

    cudaMemcpy(h_C, d_C, vectorBytesSize, cudaMemcpyDeviceToHost);

    cudaEventRecord(parallel_stop);
    cudaEventSynchronize(parallel_stop);
    cudaEventElapsedTime(&parallel_duration, parallel_start, parallel_stop);
    // ========================

    bool is_correct = true;
    for (int i = 0; i < vectorSize; i++)
        if (abs(h_C[i] - ref_C[i]) > epsilon) {
            cout << "Status: \x1B[31mWrong:\033[0m i=" << endl;
            cout << "i=" << i << " | " << h_C[i] << " != " << ref_C[i] << endl;
            is_correct = false;
            break;
        }

    if (is_correct) {
        cout << "Status: \x1B[32mCorrect\033[0m" << endl;

        cout << "Elapsed time (Sequent CPU) = " << sequent_duration << "ms" << endl;
        cout << "Elapsed time (Parallel GPU pure) = " << parallel_duration_pure << "ms" << endl;
        cout << "Elapsed time (Parallel GPU) = " << parallel_duration << "ms" << endl << endl;

        cout << "Pure acceleration: " << sequent_duration/parallel_duration_pure << endl;
        cout << "Acceleration: " << sequent_duration/parallel_duration << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete [] h_A;
    delete [] h_B;
    delete [] h_C;
    delete [] ref_C;

    return 0;
}
