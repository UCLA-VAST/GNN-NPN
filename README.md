# GNN-NPN

This repository contains the official implementation of the paper:

> **GNN-NPN: Efficient NPN Classification for Large Functions using Graph Neural Network**
>
> Chengdi Cao, Hanyu Wang, Wan-Hsuan Lin, and Jason Cong (UCLA CS)
>
> *IEEE/ACM International Conference on Computer-Aided Design (ICCAD ’26)*

## Abstract

Two Boolean functions are considered NPN equivalent if one can be transformed into the other by negating inputs, permuting inputs, or negating the output. Efficient NPN classification is valuable in fields such as logic synthesis, quantum computing, and information theory. However, exact NPN classification demands exhaustive enumeration of all possible input permutations and negations. Its complexity grows exponentially, making it impractical for larger functions with more inputs. In this paper, we introduce the first learning-based formulation for NPN classification by recasting it as NPN action prediction. Our method represents Boolean functions as AND-Inverter Graphs (AIGs) and applies a bidirectional heterogeneous graph neural network to predict input negations, output negation, and input permutations. To mitigate the non-canonicity of logic networks, we further introduce graph rewriting techniques that reduce structural ambiguity and recover useful symmetries before prediction. Experimental results show that NPN action prediction is learnable, that the resulting canonicalization quality improves over heuristic baselines, and that on practical benchmarks our method approaches exact methods on tractable cases while scaling to functions with up to 16 inputs.

## Getting Started

> **Note:** More detailed instructions for setup, data preparation, training, and evaluation will be added soon.

## License

See [LICENSE](LICENSE).
