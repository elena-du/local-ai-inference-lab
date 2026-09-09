# 04 - CPU, GPU, and NPU

CPUs favor flexible low-latency control flow, GPUs favor parallel throughput, and NPUs favor supported neural workloads at high efficiency. Transfers, unsupported operators, shapes, precision, drivers, and graph partitioning often matter more than advertised TOPS.

**Exercise:** capture `hardware` JSON. It proves device enumeration, not execution. Identify the additional runtime evidence needed for each adapter.
