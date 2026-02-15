#include <iostream> // "это БАЗА" (с)
#include <vector>  // для динамических массивов
#include <thread> // для оптимизации через многопоток
#include <chrono> // для засекания времени
#include <cmath> // для pow()
#include "openblas\cblas.h" // взял openBLAS, а не IntelMKL по двум причинам: 1 - сайт intel.com НЕ открывается, а значит невозможно
                            // безопасно скачать "intel oneapi toolskit". 2 - компьютер, на котором писался код, имеет процессор от AMD,
                            // а значит доля производительности всё же утратится и разница между MKL и openBLAS будет несущественной (по идеи).
                            // так же openBLAS был скачан готовым бинарником через vcpkg, а не скомпилирован под процессор, поэтому доля
                            // производительности может быть утеряна.

using namespace std;

const int N = 2048; // размер квадратной матрицы
const int BLOCK = 64; // размер кусков матриц (для оптимизации)

inline double& at(vector<double>& mat, int i, int j) { 
    // т.к. в программе я использовал "плоские матрицы", то сделал для них отдельные инлайн фунции, чтобы не путаться в индексации 
    // эта перегрузка вызывается для неконстантных ссылок
    return mat[i * N + j];
}

inline const double& at(const vector<double>& mat, int i, int j) {
    // эта перегрузка вызывается для константных ссылок
    return mat[i * N + j];
}

void fillMatrix(vector<double>& mas) { // Заполнение матрицы mas случайными значениями
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            at(mas, i, j) = rand() % 100; // Числа от 0 до 99
}

void prodMatrixNaive(const vector<double>& mas1, const vector<double>& mas2, vector<double>& res) { // Линейный алгоритм из линейной алгебры
 //   fill(res.begin(), res.end(), 0); // предварительно заполняем весь итоговый массив нулями

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += at(mas1, i, k) * at(mas2, k, j);  // сначала всё суммируем, следуя по формулам перемножения матриц аля a1_1*b1_1 + a1_2*b2_1 + ...
            at(res, i, j) = sum; // ... а потом присваиваем эту сумму по нужному индексу в матрицу-результат
        }
}

void prodMatrixBLAS(const vector<double>& mas1, const vector<double>& mas2, vector<double>& res) {
    // функция с алгоритмами BLAS
    cblas_dgemm(
        CblasRowMajor, // порядок хранения (либо Row, либо Col, но Col это стиль Fortran, так что используем Row)
        CblasNoTrans,  // не транспортировать первую матрицу
        CblasNoTrans,  // не транспортировать вторую матрицу
        N, N, N,  // число строк\столбцов (т.к. матрицы квадратные, то просто везде вставляем константу N)
        1.0, // масштаб
        mas1.data(), N,   // матрица и её "ведущая размерность", но т.к. у нас не подматрицы, то используем константу N
        mas2.data(), N,
        0.0, // масштаб для старых данных из матрицы-результата. Указав "0.0" мы обнуляем все значения в матрице до нуля
        res.data(), N
    );
}

void prodMatrixOwn(const vector<double>& mas1, const vector<double>& mas2, vector<double>& res) {
    /* для оптимизации применялось несколько методов:
       
       1 - разделение матриц на блоки ("блокирование"). 
       благодаря этому компилятор может засунуть такие блоки в L1\L2 кэш процессора, который намного быстрее RAM памяти

       2 - правильный порядок циклов (i->k->j).
       в наивной функции порядок циклов i->j->k, из-за чего происходят промахи кэша при прочтении mas2[k][j]. Изменив порядок
       с i->j->k на i->k->j наши данные чаще остаются в кэше процессора и он делает меньше запросов в RAM

       2.1 - переменная-буфер (buf).
       позволяет уменьшить количество обращений к памяти.
       
       3 - использование одномерного массива.
       т.к. данные лежат не фрагментами, а непрерывно, то они хорошо подходят для кэширования 

       4 - векторизация цикла j (оптимизация компилятора)

       5 - многопоточность (через openMP)
    */
    fill(res.begin(), res.end(), 0);

    //#pragma omp parallel for
    for (int ii = 0; ii < N; ii += BLOCK)
        for (int jj = 0; jj < N; jj += BLOCK)
            for (int kk = 0; kk < N; kk += BLOCK)
                for (int i = ii; i < min(N, BLOCK + ii); ++i)
                    for (int k = kk; k < min(N, BLOCK + kk); ++k) {   
                        double buf = at(mas1, i, k);
                        for (int j = jj; j < min(N, BLOCK + jj); ++j)
                            at(res, i, j) += buf * at(mas2, k, j);
                    }
}

void outputMatrix(const vector<double>& mas, int x, int y) { // вывод нужного куска матрицы
    for (int i = 0; i < x; ++i) {
        cout << endl;
        for (int j = 0; j < y; ++j)
            cout << at(mas, i, j) << " ";
    }
    cout << endl;
}

int main() {
    cout << "Created by Kalinin Timofei Nikolaevich\n" << "Code of Group: 090304-RPIa-o25\n\n";

    // создаём плоские матрицы
    vector<double> a(N * N);
    vector<double> b(N * N);
    vector<double> c1(N * N); // матрица для наивной функции
    vector<double> c2(N * N); // матрица для BLAS
    vector<double> c3(N * N); // матрица для оптимизированной функции

    srand(time(NULL)); // семечко рандома через время

    long long int diff = 2LL * N * N * N; // Сложность алгоритма

    cout << "Difficulty of Algorithm is " << diff << "\n\n";

    fillMatrix(a);
    fillMatrix(b); 

    auto start = chrono::high_resolution_clock::now();
    prodMatrixNaive(a, b, c1);
    auto end = chrono::high_resolution_clock::now();
    double time_native = chrono::duration<double>(end - start).count();
    long double perf = (diff / time_native) * pow(10, -6);
    cout << "Native Algorithm: " << time_native << " seconds\nPerformance: " << perf << " MFlops\n\n";

    start = chrono::high_resolution_clock::now();
    prodMatrixBLAS(a, b, c2);
    end = chrono::high_resolution_clock::now();
    double time_blas = chrono::duration<double>(end - start).count();
    perf = (diff / time_blas) * pow(10, -6);
    cout << "BLAS: " << time_blas << " seconds\nPerformance: " << perf << " MFlops\n\n";

    start = chrono::high_resolution_clock::now();
    prodMatrixOwn(a, b, c3);
    end = chrono::high_resolution_clock::now();
    double time_own = chrono::duration<double>(end - start).count();
    perf = (diff / time_own) * pow(10, -6);
    cout << "Own Optimization: " << time_own << " seconds\nPerformance: " << perf << " MFlops\n\n";

    if (c2 == c3 && c1 == c3)
        cout << "All Matrixs match!\n\n\n";

    // вывожу кусочки матриц для дополнительного визуального сравнения
    outputMatrix(c1, 2, 2);
    outputMatrix(c2, 2, 2);
    outputMatrix(c3, 2, 2);

    return 0;

}
