## 算法原理
[快速傅里叶变换 - 丙乙日寄。 (bycore.net)](https://bycore.net/245/FFT_mpi.html)

## 各线程蝴蝶操作的时机

  设计CUDA程序需要明确数据在kernel中的位置，以及每个线程执行操作的时机，有时还需要考虑线程之间的同步。
  前面已经讲述了并行分治的每次循环分别有哪些进程可以进入基本代码。以n=8为例，用以下表格对比**数列求和**和**FFT**操作的**第一次循环**中需要进行合并操作的两项：

| 线程 | 数列求和操作                   | FFT操作                                                      |
| ---- | ------------------------------ | ------------------------------------------------------------ |
| 0    | **nums[0]**和**nums[1]**       | **nums[BinaryReverse(0)]**和**nums[BinaryReverse(1)]**       |
| 2    | **nums[2]**和**nums[3]**       | **nums[BinaryReverse(2)]**和**nums[BinaryReverse(3)]**       |
| 4    | **nums[4]**和**nums[5]**       | **nums[BinaryReverse(4)]**和**nums[BinaryReverse(5)]**       |
| 6    | **nums[6]**和**nums[7]**       | **nums[BinaryReverse(6)]**和**nums[BinaryReverse(7)]**       |
| tid  | **nums[tid]**和**nums[tid+1]** | **nums[BinaryReverse(tid)]**和**nums[BinaryReverse(tid+1)]** |

通过对比第一次循环可以看出，定义了**BinaryReverse**操作之后，**确认合并对象在数列中的位置**这个问题已经解决，只需用BinaryReverse(tid)替代tid即可确认数的位置。
  数列求和每次合并操作合并的是两个数，但FFT每次合并操作合并的都是两个数列。为了便于理解，仍然以![n=8](https://math.jianshu.com/math?formula=n%3D8)为例，用以下表格对比**数列求和**和**FFT**操作的**第二次和第三次循环**中需要进行合并操作的两项：

| 线程 | 数列求和操作                   | FFT操作                                                      |
| ---- | ------------------------------ | ------------------------------------------------------------ |
| 0    | **nums[0]**和**nums[2]**       | 1. **nums[BinaryReverse(0)]**和**nums[BinaryReverse(2)]**2. **nums[BinaryReverse(1)]**和**nums[BinaryReverse(3)]** |
| 4    | **nums[4]**和**nums[6]**       | 1. **nums[BinaryReverse(4)]**和**nums[BinaryReverse(6)]** 2. **nums[BinaryReverse(5)]**和**nums[BinaryReverse(7)]** |
| tid  | **nums[tid]**和**nums[tid+2]** | 1. **nums[BinaryReverse(tid)]**和**nums[BinaryReverse(tid+2)]** 2. **nums[BinaryReverse(tid+1)]**和**nums[BinaryReverse(tid+3)]** |

| 线程 | 数列求和操作                   | FFT操作                                                      |
| ---- | ------------------------------ | ------------------------------------------------------------ |
| 0    | **nums[0]**和**nums[4]**       | 1. **nums[BinaryReverse(0)]**和**nums[BinaryReverse(4)]** 2. nums[BinaryReverse(1)]**和**nums[BinaryReverse(5)]** 3. nums[BinaryReverse(2)]**和**nums[BinaryReverse(6)]** 4. **nums[BinaryReverse(3)]**和**nums[BinaryReverse(7)]** |
| tid  | **nums[tid]**和**nums[tid+4]** | 1. **nums[BinaryReverse(tid)]**和**nums[BinaryReverse(tid+4)]** 2. **nums[BinaryReverse(tid+1)]**和**nums[BinaryReverse(tid+5)]** 3. **nums[BinaryReverse(tid + 2)]**和**nums[BinaryReverse(tid+6)]** 4. **nums[BinaryReverse(tid + 3)]**和**nums[BinaryReverse(tid+7)]** |

通过以上对比可以看出，在FFT操作中，tid和tid+2 (或tid+4) 的作用是进行**定位**，找到需要合并的两个数列的开头。在本文第一份代码的基础上引入一层内部循环，可以达到这个要求：

``````c
// 蝴蝶操作, 输出结果直接覆盖原存储单元的数据, factor是旋转因子
__device__ void Bufferfly(Complex *a, Complex *b, Complex factor) {
    Complex a1 = (*a) + factor * (*b);
    Complex b1 = (*a) - factor * (*b);
    *a = a1;
    *b = b1;
}

__global__ void FFT(Complex nums[], int n, int bits) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return; 
    for (int i = 2; i < 2 * n; i *= 2) {
        if (tid % i == 0) {
            // 新引入的循环
            for (int j = 0; j < k / 2; ++j) {
                Bufferfly(&nums[BinaryReverse(tid + j, bits)], &nums[BinaryReverse(tid + j + k / 2, bits)], Complex::W(k, j));
            }
        }
        __syncthreads();
    }
}
``````

上述代码已经完成了FFT运算，但FFT结果数列的顺序是按照逆转二进制数的顺序排列的，因此在拷贝结果时应按逆转二进制数的顺序来拷贝。为此，引入一个数组result来存储最终结果：

``````c
__global__ void FFT(Complex nums[], Complex result[], int n, int bits) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return;
    for (int i = 2; i < 2 * n; i *= 2) {
        if (tid % i == 0) {
            for (int j = 0; j < k / 2; ++j) {
                Bufferfly(&nums[BinaryReverse(tid + j, bits)], &nums[BinaryReverse(tid + j + k / 2, bits)], Complex::W(k, j));
            }
        }
        __syncthreads(); 
    }
    result[tid] = nums[BinaryReverse(tid, bits)];  // 拷贝到result中的对应地址
}
``````

