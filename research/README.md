# Research

These are the reports that started tierlm, written on 2026-10-01 and 2026-10-02 and exported to
Markdown. The first one was written as a Claude Doc
(https://claude.ai/code/artifact/22c33dcc-0c53-416e-9354-3d46d2cf57e6). The files here are the
copies of record for the repository.

Read them in order. Each one builds on the one before.

| # | Report | What it settles |
| --- | --- | --- |
| 01 | [Top 5 review and C implementation plan](01-review-and-plan.md) | The landscape of CPU/RAM/NVMe MoE engines and the top five (colibri, ds4, KTransformers, flash-moe, Pulsar), with pros and cons. Also the engine spec, file format, expert store, kernels, testing, a six-milestone plan and the risks. |
| 02 | [WRX90 build profile](02-wrx90-build-profile.md) | Defaults for the target box: Threadripper PRO 9965WX, 128 GB DDR5, 4× NVMe, RHEL 10.2, dual R9700. Covers build flags, threads per CCD, io_uring/O_DIRECT, KV on ≤ 10% of GPU VRAM, MTP speculation choice and Q5/Q6 quantisation. |
| 03 | [Two-stage design](03-two-stage-design.md) | The compiler (`tlmc`) vs runtime (`tierlm`) split, the nine compiler passes, the packet format, placement on four 256 GB drives, pruning and gates, and the `/etc/tierlm` setup tool. |
| 04 | [Decision engine and content selection](04-decision-engine.md) | The embedded Jev-style typed decision engine (Choice / Score / Noul), its confidence policy and audit log. Also how content exclusions in setup turn into pruning. |

Later design work is in `docs/design/`:

- [cluster.md](../docs/design/cluster.md) covers the master plus prediction slaves (MTP and
  expert pre-prediction on slaves, verified on the master), expert hosts, and other classic and
  novel arrangements.
- [swarm.md](../docs/design/swarm.md) asks whether many slow hosts, even over VPN, can give more
  throughput, and how to turn that into better reasoning.

[SOURCES.md](SOURCES.md) lists every external source cited, grouped by topic.

## Caveats

- Numbers in these reports come from the cited sources or are engineering estimates. The reports
  mark which is which. None were measured on the WRX90.
- Star counts, release versions and model names are as of early October 2026.
- Where a report and the code disagree, the code and `docs/STATUS.md` are current. The reports
  record the reasoning at the time they were written.
