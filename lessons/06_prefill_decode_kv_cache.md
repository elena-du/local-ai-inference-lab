# 06 - Prefill, decode, and KV cache

Prefill processes the prompt, often with high parallelism. Decode produces tokens iteratively and reuses keys/values from the KV cache. TTFT includes work before the first streamed token; decode throughput measures a different phase. Context and cache precision affect memory and speed.

**Exercise:** benchmark short and long-context workloads. Keep model/settings fixed and compare TTFT separately from decode tokens/sec.
