#include "mpi.h"
#include "math.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

int main(int argc, char* argv[]) {
    // MPI set up
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // the following assumes size = p x p
    int p = (int) sqrt(size);

    // Matrix set up
    srand(time(NULL));
    int n = atoi(argv[1]);
    double* my_A = new double[n * n];
    double* my_B = new double[n * n];
    double* big_A;
    double* big_B;
    double* formatted_A;
    double* formatted_B;
    if (rank == 0) { // Rank 0 generate A and B and shares them with the world
        formatted_A = new double [n*p * n*p]();
        formatted_B = new double [n*p * n*p]();
        for (int i = 0; i < n * p; i++) {
            for (int j = 0; j < n * p; j++) {
                formatted_A[i * n * p + j] = 1 + rand() % 3;
                formatted_B[i * n * p + j] = 1 + rand() % 3;
            }
        }

        big_A = new double[n * p * n * p];
        big_B = new double[n * p * n * p];
        for (int i = 0; i < p; i++){
            for (int j = 0; j < p; j++){

                for (int ii = 0; ii < n; ii++){
                    for (int jj = 0; jj < n; jj++){
                        big_A[(i*p + j)*n*n + ii*n + jj] = formatted_A[(ii + i*n)*n*p + j*n + jj];
                        big_B[(i*p + j)*n*n + ii*n + jj] = formatted_B[(ii + i*n)*n*p + j*n + jj];
                    }
                }
            }
        }
    }
    MPI_Scatter(big_A, n*n, MPI_DOUBLE, my_A, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(big_B, n*n, MPI_DOUBLE, my_B, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* C = new double [n*n]();
    double* their_A = new double [n*n];
    double* their_B = new double [n*n];

    // Jesse's collectives set up
    int row_color = rank / p;
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);

    int col_color = rank % p;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);

    // preshifting
    for (int ii = 0; ii < row_color; ii++){
        MPI_Request* A_odd = new MPI_Request;
        MPI_Request* A_even = new MPI_Request;
        int A_receiver = (p + rank - 1) % p + rank / p * p;
        int A_sender = (p + rank + 1) % p + rank / p * p;
        if (rank % 2 == 0) {
            MPI_Isend(my_A, n*n, MPI_DOUBLE, A_receiver, 0, MPI_COMM_WORLD, A_odd);
            MPI_Irecv(their_A, n*n, MPI_DOUBLE, A_sender, MPI_ANY_TAG, MPI_COMM_WORLD, A_even);
        }
        else{
            MPI_Isend(my_A, n*n, MPI_DOUBLE, A_receiver, 0, MPI_COMM_WORLD, A_even);
            MPI_Irecv(their_A, n*n, MPI_DOUBLE, A_sender, MPI_ANY_TAG, MPI_COMM_WORLD, A_odd);
        }
        MPI_Barrier(row_comm);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                my_A[i*n + j] = their_A[i*n + j];
            }
        }
    }
    for (int jj = 0; jj < col_color; jj++){
        MPI_Request* B_odd = new MPI_Request;
        MPI_Request* B_even = new MPI_Request;
        int B_receiver = (p*p + rank - p) % (p*p);
        int B_sender = (p*p + rank + p) % (p*p);
        if (rank / 2 == 0) {
            MPI_Isend(my_B, n*n, MPI_DOUBLE, B_receiver, 0, MPI_COMM_WORLD, B_odd);
            MPI_Irecv(their_B, n*n, MPI_DOUBLE, B_sender, MPI_ANY_TAG, MPI_COMM_WORLD, B_even);
        }
        else{
            MPI_Isend(my_B, n*n, MPI_DOUBLE, B_receiver, 0, MPI_COMM_WORLD, B_even);
            MPI_Irecv(their_B, n*n, MPI_DOUBLE, B_sender, MPI_ANY_TAG, MPI_COMM_WORLD, B_odd);
        }
        MPI_Barrier(col_comm);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                my_B[i*n + j] = their_B[i*n + j];
            }
        }
    }

    // Canon's matmul
    for (int block = 0; block < p; block ++) {
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    C[i * n + j] += my_A[i * n + k] * my_B[k * n + j];
                }
            }
        }
        // Communicate
        if (block < p - 1) {
            MPI_Request* A_odd = new MPI_Request;
            MPI_Request* A_even = new MPI_Request;
            MPI_Request* B_odd = new MPI_Request;
            MPI_Request* B_even = new MPI_Request;
            int A_receiver = (p + rank - 1) % p + rank / p * p;
            int A_sender = (p + rank + 1) % p + rank / p * p;
            int B_receiver = (p*p + rank - p) % (p*p);
            int B_sender = (p*p + rank + p) % (p*p);
            if (rank % 2 == 0) {
                MPI_Isend(my_A, n*n, MPI_DOUBLE, A_receiver, 0, MPI_COMM_WORLD, A_odd);
                MPI_Irecv(their_A, n*n, MPI_DOUBLE, A_sender, MPI_ANY_TAG, MPI_COMM_WORLD, A_even);
            }
            else{
                MPI_Isend(my_A, n*n, MPI_DOUBLE, A_receiver, 0, MPI_COMM_WORLD, A_even);
                MPI_Irecv(their_A, n*n, MPI_DOUBLE, A_sender, MPI_ANY_TAG, MPI_COMM_WORLD, A_odd);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank / 2 == 0) {
                MPI_Isend(my_B, n*n, MPI_DOUBLE, B_receiver, 0, MPI_COMM_WORLD, B_odd);
                MPI_Irecv(their_B, n*n, MPI_DOUBLE, B_sender, MPI_ANY_TAG, MPI_COMM_WORLD, B_even);
            }
            else{
                MPI_Isend(my_B, n*n, MPI_DOUBLE, B_receiver, 0, MPI_COMM_WORLD, B_even);
                MPI_Irecv(their_B, n*n, MPI_DOUBLE, B_sender, MPI_ANY_TAG, MPI_COMM_WORLD, B_odd);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            for (int i = 0; i < n; i++){
                for (int j = 0; j < n; j++){
                    my_A[i*n + j] = their_A[i*n + j];
                    my_B[i*n + j] = their_B[i*n + j];
                }
            }
        }
    }

    // Rank zero gathering information back
    double* big_C;
    if (rank == 0){
        big_C = new double[n*p * n*p];
    }
    MPI_Gather(C, n*n, MPI_DOUBLE, big_C, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){
        // Compare with the real matmul
        double * real_C = new double [n*p * n*p]();
        for (int i = 0; i < n*p; ++i){
            for (int k = 0; k < n*p; ++k){
                for (int j = 0; j < n*p; ++j){
                    real_C[j + i * n * p] += formatted_A[k + i * n * p] * formatted_B[j + k * n * p];
                }
            }
        }

        double error = 0;
        for (int i = 0; i < n*p; i++){
            for (int j = 0; j < n*p; j++){
                error += real_C[i*n*p + j] - big_C[i*n*p + j];
            }
        }
        cout << "Error is " << error << endl;

        delete[] real_C;
        delete[] big_A;
        delete[] big_B;
        delete[] big_C;
        delete[] formatted_A;
        delete[] formatted_B;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    delete[] my_A;
    delete[] my_B;
    delete[] their_A;
    delete[] their_B;
    delete[] C;

    MPI_Finalize();
}
