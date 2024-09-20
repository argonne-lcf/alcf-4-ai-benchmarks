# Collective communication benchmark

./benchmark.py is a wrapper code which will perform different communication benchmarks with different backend. It utizes other communication benchmarks whenever it is possible, including OSU benchmark, NCCL benchmark, RCCL benchmark, and oneCCL benchmark. 

```bash
./benchmark.py
  --api: [direct|ddp|hvd|mpi4py]
  --coll: [allreduce|allgather|bcast|alltoall]
  --iters: number of iterations 
  --min_elem_count: min messize size 
  --max_elem_count: max messize size
  --warmup_iters: number of warmup iterations
  --comm_lib [mpi|nccl|rccl|ccl]
```

In the output, it provides the latency, bandwidth for different message sizes specified.
