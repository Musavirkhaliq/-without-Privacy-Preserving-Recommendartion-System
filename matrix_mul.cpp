#include <iostream>

class Matrix {
private:
    int rows, cols;
    int** data;

public:
    // Constructor
    Matrix(int r, int c, int** d) : rows(r), cols(c) {
        data = new int*[rows];
        for (int i = 0; i < rows; i++) {
            data[i] = new int[cols];
        }
        for(int i=0; i<r; i++)
            for(int j=0; j<c; j++)
                data[i][j]=d[i][j];
    }

    // Destructor
    ~Matrix() {
        for (int i = 0; i < rows; i++) {
            delete[] data[i];
        }
        delete[] data;
    }

    // Copy constructor
    Matrix(const Matrix &m) {
        rows = m.rows;
        cols = m.cols;
        data = new int*[rows];
        for (int i = 0; i < rows; i++) {
            data[i] = new int[cols];
            for (int j = 0; j < cols; j++) {
                data[i][j] = m.data[i][j];
            }
        }
    }

    // Assignment operator
    Matrix& operator=(const Matrix &m) {
        if (this != &m) {
            // Free existing memory
            for (int i = 0; i < rows; i++) {
                delete[] data[i];
            }
            delete[] data;

            rows = m.rows;
            cols = m.cols;
            data = new int*[rows];
            for (int i = 0; i < rows; i++) {
                data[i] = new int[cols];
                for (int j = 0; j < cols; j++) {
                    data[i][j] = m.data[i][j];
                }
            }
        }
        return *this;
    }
    // matrix addition
    Matrix operator+(const Matrix &m) {
        if(rows != m.rows || cols != m.cols) {
            std::cout << "Error: Matrix addition with different dimensions\n";
            return *this;
        }
        Matrix result(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + m.data[i][j];
            }
        }
        return result;
    }
    // matrix subtraction
    Matrix operator-(const Matrix &m) {
        if(rows != m.rows || cols != m.cols) {
            std::cout << "Error: Matrix subtraction with different dimensions\n";
            return *this;
        }
        Matrix result(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - m.data[i][j];
            }
        }
        return result;
    }

    // matrix multiplication
    Matrix operator*(const Matrix &m) {
        if (cols != m.rows) {
            std::cout << "Error: Cannot perform matrix multiplication. Number of columns of the first matrix must be equal to the number of rows of the second matrix." << std::endl;
            return *this;
        }

        Matrix result(rows, m.cols);

        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                result.data[i][j] = 0;
                for (int k = 0; k < cols; k++) {
                    result.data[i][j] += data[i][k] * m.data[k][j];
                }
            }
        }
        return result;
    }

    // print the matrix
    void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

int main() {
    int d1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Matrix m1(2, 3, (int**)d1);
    int d2[3][2] = {{7, 8}, {9, 10}, {11, 12}};
    Matrix m2(3, 2, (int**)d2);

    Matrix m3 = m1 * m2;
    m3.print();

    return 0;
}
