
class CudaKernelLock {
private:
    int* mutex;
public:
    CudaKernelLock(void) {
        int state = 0;
        cudaError_t ret = cudaMalloc((void**)&mutex, sizeof(int));
        ret = cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    }
    ~CudaKernelLock() {
        cudaFree(mutex);
    }

    __device__ void lock(void) {
        while (atomicCAS(mutex, 0, 1) != 0);
    }
    __device__ void unlock(void) {
        atomicExch(mutex, 0);
    }
};

//多对一reduce
__device__ float atomic_log_plus(float *addr_f, float value) {
    // addr_f 为公有变量地址，所有的线程均要把结果累加到该地址
    // 浮点指针转化为整型指针，因为atomicCAS只接受整型指针 
    int *addr = (int*)addr_f;
    // 读取addr_f处的值，姑且成为addr_f初始值
    float expected = *addr_f;
    // 计算addr_f初始值与value求和的结果
    float sum = log_plus(expected, value);
    // 判断在求和期间addr_f地址有没有被别的线程修改过
    // 如果addr_f未被修改过，则addr处的值与expected相同，sum 赋值给addr, 
    //    old_value 为addr 处的值，也应与expected 相同（下面while loop 会用到）
    // 如果addr_f被修改过，则addr处的值必然与expected 不同。
    //    则addr地址对应的值不变，并将该值赋给old_value
    int old_value = atomicCAS(addr, __float_as_int(expected), __float_as_int(sum));
    
    // 如果old_value 与expected 不同，说明addr_f 已被别的线程修改，
    // 上述sum 计算无效，当前线程未能将value 叠加到addr_f上，
    // 进入while loop重新尝试将value 叠加到addr_f上
    while (old_value != __float_as_int(expected)) {
        // 给expected 赋值未old_value, 
        // (此old_value 一点也不old, 它是别的线程写addr_f处后该地址对应的最新的值)
        expected = __int_as_float(old_value);
        // 再一次把value累加到expected， 即addr_f地址处最新的值上
        sum = log_plus(expected, value);
        // 再一次判断计算sum期间 addr_f有没有被别的线程写入新值
        // 如果addr_f没有被别的线程写入，把sum值写入到addr, 即addr_f
        // 如果addr_f又被别的线程写入了，则再执行一趟whill iteration, 
        // 直到当前线程成功将sum写入为止
        old_value = atomicCAS(addr, __float_as_int(expected), __float_as_int(sum));
    }
    return __int_as_float(old_value);