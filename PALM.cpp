#include <iostream>
#include <cmath>

using namespace std;
int m = 5, n = 6, k = 2;
double *X = new double[m*k];
double *Y = new double[k*n];


//printing the matrix
void print_matrix(double *Z,int m,int n) {
    cout << "The final matrix Z is :" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << Z[i*n + j] << " ";
        }
        cout << endl;
    }
}
    

//l2_norm computation
double l2_norm(double *X, double *Y, int m, int n) {
    double sum = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff = X[i*n + j] - Y[i*n + j];
            sum += diff * diff;
        }
    }
    return sqrt(sum);
}

//transpose the matrix inplace
double *transpose_matrix(double *matrix, int m, int n) {
    double *transposed_matrix = new double[n*m];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            transposed_matrix[j*n + i] = matrix[i*n + j];
        }
    }
    return transposed_matrix;
}



void matrix_factorization_PALM(double *R, int m, int n, int k, double lambda_1=0.1,double lambda_2=0.2, int max_iter=100000, double tol=1e-4) {
    

    //Initialize X and Y with random values
    for (int i = 0; i < m*k; i++) {
        X[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < k*n; i++) {
        Y[i] = (double)rand() / RAND_MAX;
    }

    // Set the step size
    double norm_X = 0, norm_Y = 0;
    for (int i = 0; i < m*k; i++) {
        norm_X += X[i] * X[i];
    }
    for (int i = 0; i < k*n; i++) {
        norm_Y += Y[i] * Y[i];
    }
    double eta = 1 / (sqrt(norm_X) * sqrt(norm_Y));

    for (int iter = 0; iter < max_iter; iter++) {
        // Minimize with respect to X
        double *YYt = new double[k*k];
        double *RYt = new double[m*k];
        double L = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                double temp = 0;
                for (int p = 0; p < n; p++) {
                    temp += Y[p*k + i] * Y[p*k + j];
                }
                YYt[i*k + j] = temp;
                L = max(L, temp);
            }
        }
        L = max(L, 1e-4);
        eta = 1/(1.1*L);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double temp = 0;
                for (int p = 0; p < n; p++) {
                    temp += R[i*n + p] * Y[p*k + j];
                }
                RYt[i*k + j] = 5*temp;
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double temp = 0;
                for (int p = 0; p < k; p++) {
                    temp += X[i*k + p] * YYt[p*k + j];
                }
                X[i*k + j] = X[i*k + j] - eta * (temp - RYt[i*k + j]);
            }
        }
        delete[] YYt;
        delete[] RYt;

        // Minimize with respect to Y
        double *XtX = new double[k*k];
        double *RtX = new double[k*n];
        L = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                double temp = 0;
                for (int p = 0; p < m; p++) {
                    temp += X[p*k + i] * X[p*k + j];
                }
                XtX[i*k + j] = temp;
                L = max(L, temp);
            }
        }
        L = max(L, 1e-4);
        eta = 1/(1.1*L);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                double temp = 0;
                for (int p = 0; p < m; p++) {
                    temp += R[p*n + j] * X[p*k + i];
                }
                RtX[i*n + j] = temp;
            }
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                double temp = 0;
                for (int p = 0; p < k; p++) {
                    temp += Y[p*n + j] * XtX[i*k + p];
                }
                Y[i*n + j] = Y[i*n + j] - eta * (temp - RtX[i*n + j]);
            }
        }
        delete[] XtX;
        delete[] RtX;
    }
}

int main()
{
int m = 5, n = 6, k = 2;
double R[5*6] = {1, 3, 4, 2, 5, 0,
                 1, 4, 5, 3, 2, 4, 
                 5, 0, 2, 1, 4, 5, 
                 3, 0, 1, 5, 2, 4, 
                 3, 5, 1, 0, 2, 4}; //initialize this with the values of the input matrix

print_matrix(R, m, n);
double *transposed_matrix = transpose_matrix(R, m, n);
print_matrix(transposed_matrix, n, m);
/*
matrix_factorization_PALM(R, m, n, k);
print_matrix(X, m, k);

// printing the factor matrices X and Y as needed
// for (int i = 0; i < m; i++) {
//     for (int j = 0; j < k; j++) {
//         cout << X[i*k + j] << " ";
//     }
//     cout << endl;
// }
// cout << endl;
print_matrix(Y, k, n);
// for (int i = 0; i < k; i++) {
//     for (int j = 0; j < n; j++) {
//         cout << Y[i*n + j] << " ";
//     }
//     cout << endl;
// }



double *Z = new double[m*n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double temp = 0;
            for (int p = 0; p < k; p++) {
                temp += X[i*k + p] * Y[p*n + j];
            }
            Z[i*n + j] = temp;
        }
    }

print_matrix(Z, m, n);
//     cout << "The final matrix Z is :" << endl;
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             cout << Z[i*n + j] << " ";
//         }
//         cout << endl;
//     }
cout<<l2_norm(X,Y,m,n);
    delete[] X;
    delete[] Y;
    delete[] Z;
    return 0;*/
}