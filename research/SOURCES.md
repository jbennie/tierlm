# Sources

This page lists every external source cited in `research/`, `docs/design/` and `fragments/`,
grouped by topic. All of them were accessed between 2026-10-01 and 2026-10-02. The "Cited in"
column uses these short names:

| Short name | File |
| --- | --- |
| 01 | `research/01-review-and-plan.md` |
| 02 | `research/02-wrx90-build-profile.md` |
| 03 | `research/03-two-stage-design.md` |
| 04 | `research/04-decision-engine.md` |
| cluster | `docs/design/cluster.md` |
| swarm | `docs/design/swarm.md` |

The reference engines were read as source code. The fragments summarise them in our own words
(see `NOTICE` for their licences).

## Reference engines (read as source)

| Project | Tag | Licence | Cited in |
| --- | --- | --- | --- |
| [JustVugg/colibri](https://github.com/JustVugg/colibri) | colibri | Apache-2.0 | 01, 02, 03, 04, fragments |
| [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4) and its [SSD streaming guide](https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md) | ds4 | MIT | 01, cluster, fragments |
| [giannisanni/pulsar](https://github.com/giannisanni/pulsar) | pulsar | MIT | 01, 03, fragments |
| [Pioshin/monolite](https://github.com/Pioshin/monolite) | monolite | Apache-2.0 | 01, fragments |
| [Schero94/slipstream](https://github.com/Schero94/slipstream) | slipstream | MIT | 01, 03, fragments |
| [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | ktransformers | Apache-2.0 | 01, fragments |
| [bricesommers/apus-glm5.3-flash](https://github.com/bricesommers/apus-glm5.3-flash) | apus | PolyForm-NC-1.0.0 | 01, 02, fragments |
| [danveloper/flash-moe](https://github.com/danveloper/flash-moe) | flashmoe | none | 01, fragments |

## Landscape, news and articles

- [LMSYS: SGLang SSD Expert Pack](https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/)
  (cited in 01, 03)
- [LMSYS: KTransformers in SGLang](https://www.lmsys.org/blog/2025-10-22-KTransformers)
  (cited in 01)
- [Phoronix on KTransformers 0.7](https://phoronix.com/news/KTransformers-0.7) and
  [KTransformers 0.5.3 AVX2](https://www.phoronix.com/news/KTransformers-0.5.3) (cited in 01)
- [llama.cpp discussion #27149: expert-aware SSD streaming](https://github.com/ggml-org/llama.cpp/discussions/27149)
  (cited in 01)
- [GitHub topic: ssd-streaming](https://github.com/topics/ssd-streaming) (cited in 01)
- [InsiderLLM: Your SSD is the new GPU](https://buttondown.com/insiderllm/archive/your-ssd-is-the-new-gpu/)
  (cited in 01)
- [moe-stream on Jetson: a 26B MoE on 8 GB by streaming experts](https://dev.to/anyesh/running-a-26b-moe-on-an-8-gb-jetson-by-streaming-experts-from-ssd)
  (cited in 01)
- [gmlx on PyPI](https://pypi.org/project/gmlx/) (cited in 01)

## Speculative decoding and MTP

- [AcceptMoE (arXiv 2608.02989)](https://arxiv.org/abs/2608.02989) (cited in 01, 02)
- [DFlash (arXiv 2602.06036)](https://arxiv.org/html/2602.06036v2) (cited in 01, 02)
- [Bigger draft trees hurt on MoE: EAGLE-3 vs MTP on GLM-4.7-Flash](https://medium.com/@ineshreddy249/bigger-draft-trees-hurt-on-moe-benchmarking-eagle3-vs-mtp-on-glm-4-7-flash-74591c39a873)
  (cited in 02)
- [vLLM: speculative decoding on AMD GPUs](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)
  (cited in 02)
- Across devices (all cited in cluster):
  - [DSI, Distributed Speculative Inference](https://arxiv.org/abs/2405.14105)
  - [PipeInfer](https://arxiv.org/abs/2407.11798)
  - [FlowSpec](https://arxiv.org/abs/2507.02620)
  - [SpecExec](https://arxiv.org/abs/2406.02532)
  - [Lookahead decoding](https://arxiv.org/abs/2402.02057)
  - [SpecEdge](https://arxiv.org/abs/2505.17052)
  - [SLED](https://arxiv.org/abs/2506.09397)
  - [SwiftSpec](https://arxiv.org/abs/2506.11309)
  - [Mirror Speculative Decoding](https://arxiv.org/abs/2510.13161)

## MoE pruning, expert prediction, placement and caching

- [REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression](https://arxiv.org/abs/2510.13999v1)
  and [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) (cited in 03, 04)
- Cited in cluster:
  - [SP-MoE](https://arxiv.org/abs/2510.10302)
  - [MoE-SpeQ](https://arxiv.org/abs/2511.14102)
  - [OD-MoE](https://arxiv.org/abs/2512.03927)
  - [Speculative MoE](https://arxiv.org/abs/2503.04398)
  - [Pre-gated MoE (ISCA 2024)](https://arxiv.org/abs/2308.12066)
  - [MoE-Infinity](https://arxiv.org/abs/2401.14361)
  - [AdapMoE](https://arxiv.org/abs/2408.10284)
  - [SlimCaching](https://arxiv.org/abs/2507.06567)

## Decision models (Jev style)

- [CometAPI: Jev model overview (TypeSafe AI)](https://www.cometapi.com/models/typesafe-ai/jev)
  (cited in 04)
- [OpenJevX on Hugging Face](https://huggingface.co/muthuishere/openjevx) (cited in 04)
- [Laya model family](https://www.llmreference.com/model-family/laya) and
  [Laya explained](https://www.orcarouter.ai/blog/laya-decision-model-explained) (cited in 04)
- [Type-Safe Is Not Error-Free (arXiv 2609.26758)](https://arxiv.org/pdf/2609.26758) (cited in 04)

## Target hardware and OS

- [ASUS Pro WS WRX90E-SAGE SE specifications](https://www.asus.com/us/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/techspec/)
  and the [product page (Intel X710-AT2)](https://www.asus.com/no/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/)
  (cited in 02, cluster)
- [TechSpot: Threadripper 9000 review (CCD counts)](https://www.techspot.com/review/3020-amd-ryzen-threadripper-9980x-9970x/)
  (cited in 02)
- [Level1Techs: When is 8 memory channels not 8 memory channels?](https://forum.level1techs.com/t/when-is-8-memory-channels-not-8-memory-channels/239685)
  (cited in 02)
- [AMD Radeon AI PRO R9700 Linux drivers](https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-ai-pro/radeon-ai-pro-r9000-series/amd-radeon-ai-pro-r9700.html)
  and [ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
  (cited in 02)
- [RHEL-65347: io_uring tech preview, disabled by default in RHEL 10](https://issues.redhat.com/browse/RHEL-65347)
  (cited in 02)

## Networking, RDMA and links (all cited in cluster)

- [Intel Linux RDMA driver for E810 and X722](https://www.intel.com/content/www/us/en/download/19632/29875/linux-rdma-driver-for-the-e810-and-x722-intel-ethernet-controllers.html)
- [RHEL 8.8 release notes: deprecated kernel functionality (Soft-RoCE)](https://docs.redhat.com/ja/documentation/red_hat_enterprise_linux/8/html/8.8_release_notes/deprecated-functionality-kernel)
- [RHEL 9: Configuring InfiniBand and RDMA networks](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html-single/configuring_infiniband_and_rdma_networks)
- [Level1Techs: Ryzen AI Halo USB4 clustering with RDMA](https://forum.level1techs.com/t/ryzen-ai-halo-usb4-clustering-with-rdma-testing-notes/253117)
- [CachyOS forum: ASM4242 Thunderbolt networking on Linux 7.2](https://discuss.cachyos.org/t/asm4242-thunderbolt-networking-unusable-on-linux-cachyos-7-2-0-1/34826)
- [Linux kernel: USB4 and Thunderbolt](https://docs.kernel.org/admin-guide/thunderbolt.html)
- [Gigazine: Mac Studio RDMA over Thunderbolt](https://gigazine.net/gsc_news/en/20251222-mac-rdma-over-thunderbolt-1-5tb)
- [Busy polling next generation (netdev 2.1)](https://www.netdevconf.org/2.1/papers/BusyPollingNextGen.pdf)
- [RAMCloud: InfiniBand verbs performance](https://ramcloud.atlassian.net/wiki/spaces/RAM/pages/6848596/Infiniband+Verbs+Performance)
- [NVIDIA forum: RoCEv2 performance test](https://forums.developer.nvidia.com/t/performance-test-with-rocev2/241928)
- [Kalia et al., eRPC (NSDI 2019)](https://www.usenix.org/conference/nsdi19/presentation/kalia)
- [Soft-RoCE overview (OFA 2017)](https://www.openfabrics.org/images/eventpresos/2017presentations/205_SoftRoCE_LLiss.pdf)
- [NVIDIA RDMA Aware Networks Programming manual](https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17/key+concepts)
- [rdma-core](https://github.com/linux-rdma/rdma-core) and [perftest](https://github.com/linux-rdma/perftest)
- [io_uring zero-copy receive](https://docs.kernel.org/6.15/networking/iou-zcrx.html)

## Distributed and decentralised inference

- Cited in cluster:
  - [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc)
  - [llama.cpp distributed benchmarks over 10GbE](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)
  - [exo](https://github.com/exo-explore/exo)
  - [distributed-llama](https://github.com/b4rtaz/distributed-llama)
  - [prima.cpp](https://arxiv.org/abs/2504.08791)
  - [MegaScale-Infer](https://arxiv.org/abs/2504.02263)
  - [DistServe](https://arxiv.org/abs/2401.09670)
  - [Splitwise](https://arxiv.org/abs/2311.18677)
  - [Mooncake](https://arxiv.org/abs/2407.00079)
  - [LMCache](https://arxiv.org/abs/2510.09665)
  - [vLLM NIXL connector](https://docs.vllm.ai/en/v0.25.0/features/nixl_connector_usage/)
- Petals: [arXiv 2209.01188](https://arxiv.org/abs/2209.01188) and
  [arXiv 2312.08361](https://arxiv.org/abs/2312.08361) (cited in cluster, swarm)
- [Prime Intellect: decentralized inference](https://www.primeintellect.ai/blog/inference)
  (cited in swarm)
- [Learning to reason across parallel samples](https://arxiv.org/pdf/2506.09014) and
  [ParaThinker: native parallel thinking](https://arxiv.org/pdf/2509.04475) (cited in swarm)

## Far memory, cooperative caching, tail latency (all cited in cluster)

- [Memory disaggregation: advances and open challenges](https://arxiv.org/abs/2305.03943)
- [Infiniswap (NSDI 2017)](https://rtcl.eecs.umich.edu/rtclweb/assets/publications/2017/infiniswap-nsdi17.pdf)
- [Dahlin et al., Cooperative Caching (OSDI 1994)](https://www.usenix.org/legacy/publications/library/proceedings/osdi/dahlin.html)
- [The Tail at Scale (summary)](https://blog.acolyer.org/2015/01/15/the-tail-at-scale/)
