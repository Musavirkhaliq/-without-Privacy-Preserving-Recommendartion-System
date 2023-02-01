#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;

const int N = 1e5 + 10, M = 1e5 + 10, K = 10;
const double lambda = 0.01, eps = 1e-5;
double globalBias = 0;

int n, m, k;
double A[N][K], B[M][K], C[N][M];
extern double bu[N], bi[M];
void init() {
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= k; j++) {
            A[i][j] = (rand() / (double)RAND_MAX) / k;
        }
    }
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= k; j++) {
            B[i][j] = (rand() / (double)RAND_MAX) / k;
        }
    }
    for(int i = 1; i <= n; i++)
      bu[i] = (rand() / (double)RAND_MAX) - 0.5;
    for(int i = 1; i <= m; i++)
      bi[i] = (rand() / (double)RAND_MAX) - 0.5;
}

double calc_error() {
    double res = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            double tmp = globalBias + bu[i] + bi[j];
            for (int t = 1; t <= k; t++) {
                tmp += A[i][t] * B[j][t];
            }
            res += (C[i][j] - tmp) * (C[i][j] - tmp);
        }
    }
    double bu_dot = 0, bi_dot = 0;
    for (int i = 1; i <= n; i++) {
        bu_dot += bu[i] * bu[i];
    }
    for (int i = 1; i <= m; i++) {
        bi_dot += bi[i] * bi[i];
    }

    res += lambda * (sqrt(bu_dot) + sqrt(bi_dot));
    return res;
}

void update_A() {
    for (int i = 1; i <= n; i++) {
        double tA[K];
        memset(tA, 0, sizeof(tA));
        for (int j = 1; j <= m; j++) {
            if (C[i][j] == 0) continue;
            for (int t = 1; t <= k; t++) {
                tA[t] += B[j][t] * (C[i][j] - globalBias - bu[i] - bi[j]);
            }
        }
        for (int j = 1; j <= k; j++) {
            double tB[K];
            memset(tB, 0, sizeof(tB));
            for (int t = 1; t <= m; t++) {
                if (C[i][t] == 0) continue;
                tB[j] += B[t][j] * B[t][j];
            }
            A[i][j] = tA[j] / (lambda + tB[j]);
        }
    }
}

void update_B() {
    for (int i = 1; i <= m; i++) {
        double tB[K];
        memset(tB, 0, sizeof(tB));
        for (int j = 1; j <= n; j++) {
            if (C[j][i] == 0) continue;
            for (int t = 1; t <= k; t++) {
                tB[t] += A[j][t] * (C[j][i] - globalBias - bu[j] - bi[i]);
            }
        }
        for (int j = 1; j <= k; j++) {
            double tA[K];
            memset(tA, 0, sizeof(tA));
            for (int t = 1; t <= n; t++) {
                if (C[t][i] == 0) continue;
                tA[j] += A[t][j] * A[t][j];
            }
            B[i][j] = tB[j] / (lambda + tA[j]);
        }
    }
}

void update_bu() {
    for (int i = 1; i <= n; i++) {
        double sum = 0;
        for (int j = 1; j <= m; j++) {
            if (C[i][j] == 0) continue;
            sum += C[i][j] - globalBias - bu[i] - bi[j];
        }
        bu[i] = sum / (lambda + m);
    }
}

void update_bi() {
    for (int i = 1; i <= m; i++) {
        double sum = 0;
        for (int j = 1; j <= n; j++) {
            if (C[j][i] == 0) continue;
            sum += C[j][i] - globalBias - bu[j] - bi[i];
        }
        bi[i] = sum / (lambda + n);
    }
}

void ALS() {
    init();
    double last_error = 1e9, error = calc_error();
    while (fabs(last_error - error) > eps) {
        update_A();
        update_B();
        update_bu();
        update_bi();
        last_error = error;
        // error = calc_error();
    }
}

int main() {
    cin >> n >> m >> k;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            cin >> C[i][j];
        }
    }
    ALS();
    return 0;
}

