#include <iostream>
#include <vector>

using namespace std;

// Define a Matrix class
template <typename Type>
class Matrix
{
private:
    // Number of rows and columns
    size_t rows;
    size_t cols;

    // Vector to store the elements
    vector<Type> data;

public:
    // Shape of the matrix
    tuple<size_t, size_t> shape;
    // Number of elements in the matrix
    int num_ele;

    // Constructor
    Matrix(size_t rows, size_t cols)
    {
        this->rows = rows;
        this->cols = cols;
        // Empty vector for the data
        this->data = {};
        this->data.resize(rows * cols, Type());
        this->shape = {rows, cols};
        this->num_ele = rows * cols;
    }
    Matrix()
    {
        this->rows = 0;
        this->cols = 0;
        this->data = {};
        this->shape = {rows, cols};
        this->num_ele = rows * cols;
    }

    // Print functions
    void print_shape()
    {
        cout << "Matrix Size([" << rows << ", " << cols << "])"
             << "\n";
    }
    void print()
    {
        for (size_t r = 0; r < rows; r++)
        {
            for (size_t c = 0; c < cols; c++)
            {
                cout << (*this)(r, c) << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }

    // Access Matrix elements
    Type &operator()(size_t row, size_t col)
    {
        return this->data[rows * cols + cols];
    }
};

int main()
{
    Matrix<int> M = Matrix<int>(3, 6);
    M.print_shape();
    M.print();
    cout << M(2, 2) << "\n";
    return 0;
}