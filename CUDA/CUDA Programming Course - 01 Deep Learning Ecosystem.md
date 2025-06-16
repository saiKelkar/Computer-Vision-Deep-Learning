(Main Resources: 
https://www.youtube.com/watch?v=LuhJEEJQgUM&ab_channel=GPUMODE )

- - - 
**GPUs AS PARALLEL COMPUTERS:** 

GFLOPS -- giga floating-point operations per second

![[Pasted image 20250617002856.png]]

CPUs and GPUs have fundamentally different design philosophies. 
- The design of a CPU is optimized for sequential code performance. It makes use of control logic to allow instructions from a single thread of execution to execute in parallel or even out of their sequential order while maintaining the appearance of sequential execution. 
  Large cache memories are provided to reduce the instruction and data access latencies of large complex applications. 
- For GPUs, to optimize for the execution of floating-point calculations, a massive number of threads are used. The hardware takes advantage of a large number of execution threads to find work to do when some of them are waiting for long-latency memory accesses, thus minimizing the control logic required for each execution thread. 
  Small cache memories are provided to help control the bandwidth requirements of these applications so multiple threads that access the same memory data do not need to go to the DRAM. As a result, more chip area is dedicated to the floating-point calculations. 

CUDA (Compute Unified Device Architecture) programming model -- designed to support joint CPU/GPU execution of an application. 

---

**ARCHITECTURE OF A MODERN GPU:**



