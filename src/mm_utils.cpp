#include <cstring>
// 因为本文件是 .cpp 文件, 所以推荐用 cstring 而不是 string.h, 下面的 cstdlib 同理
// 注意 c++ 的 string 类应该用 #include<string>
#include <cstdlib>
#include <random>

// argv = [./run -xyz 123], option="-xyz"  -> 2
// argv = [./run -xyz 123], option="-xy"   -> -1
int parse_cmdline(int argc, char **argv, char const *option){
    for (int i = 1; i < argc; ++i){
        if (strcmp(argv[i], option) == 0)  // strcmp 来源于 cstring 头文件
            return i + 1;
    }
    return -1;
}

int parse_cmdline_int(int argc, char **argv, char const *option, int default_value){
    int i = parse_cmdline(argc, argv, option);
    if (i > 0)
        return std::atoi(argv[i]);
    else
        return default_value;
}

void print_matrix(float *matrix, int row, int col){
    printf("[\n");
    for (int i = 0; i < row; ++i)
        {
            printf("[");
            for (int j = 0; j < col; ++j)
                printf("%f, ", matrix[i*col+j]);
            printf("]\n");
        }
    printf("]\n");
}

void random_init(float *matrix, int row, int col){
    // 创建随机数引擎
    std::random_device rd;
    std::mt19937 gen(rd()); // 使用 Mersenne Twister 算法
    // 创建-1到1之间的均匀分布
    std::uniform_real_distribution<> distribution(-1.0, 1.0);

    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            matrix[i*col+j] = distribution(gen);
}

void const_init(float *matrix, int row, int col, float value){
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            matrix[i*col+j] = value;
}

template<typename T>
T get_value(T const *M, int row, int col, int i, int j){
    return M[i*col+j];
}

template<typename T>
T set_value(T const *M, int row, int col, int i, int j, T value){
    return M[i*col+j] = value;
}

template<typename T>
T accumulate_value(T const *M, int row, int col, int i, int j, T value){
    return M[i*col+j] += value;
}
