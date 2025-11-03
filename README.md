# BSAUEA Project

## Important Notes

> **Memory Management:**
> - Unsupervised loss computation must be performed on CPU to prevent memory overflow
> - Temporal similarity matrix calculation requires CPU execution
> - For YAGO-WIKI50K dataset, switch to CPU mode if GPU memory is insufficient

## Dataset Preparation

### Heterogeneous Datasets
Download heterogeneous datasets from:  
[https://github.com/DataArcTech/Simple-HHEA](https://github.com/DataArcTech/Simple-HHEA)

### GloVe Embeddings
1. Download `glove.6B.zip` from:  
   [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
2. Unzip into `data/` directory
3. File `glove.6B.300d.txt` will be used for training

## Environment Requirements

**System Configuration:**
- **OS:** Ubuntu 18.04
- **Python:** 3.8
- **CUDA:** 11.3
- **cuDNN:** 8
- **NVCC:** Required

**Python Packages:**
- `torch == 1.11.0`
- `torchvision == 0.12.0`
- `torchaudio == 0.11.0`

## Dataset Format Specification

**File Structure:**
- `ent_ids_1`: Entity IDs of source knowledge graph
- `ent_ids_2`: Entity IDs of target knowledge graph  
- `triples_1`: Relation quadruples (ID-encoded) in source KG
- `triples_2`: Relation quadruples (ID-encoded) in target KG
- `rel_ids_1`: Relation IDs of source knowledge graph
- `rel_ids_2`: Relation IDs of target knowledge graph
- `sup_pairs + ref_pairs`: Aligned entity pairs
- `ref_ent_ids`: Combined `sup_pairs + ref_pairs`

**Special Note for YAGO-WIKI50K/WIKI50K:**  
While the original format uses Wikipedia + QID identifiers, this implementation utilizes the **entity name version** (Wikipedia + entity names).

## Acknowledgements

This project references code from:
- **RAGA**
- **Simple-HHEA**

We extend our gratitude to the authors for their valuable contributions to the research community.
