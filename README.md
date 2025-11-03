> The unsupervised loss value needs to be computed on the CPU to avoid memory overflow.
> The calculation of the temporal similarity matrix also needs to use the CPU.
> When running yogo-wiki50k, if the GPU memory is insufficient, you can try switching to CPU execution.

The heterogeneous dataset comes from: https://github.com/DataArcTech/Simple-HHEA

download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
unzip glove.6B.zip into data/ (glove.6B.300d.txt will be used)

ENVIRONMENT
  Requirements
  Ubuntu18.04,
  Python 3.8, 
  CUDA 11.3, 
  cuDNN 8, 
  NVCC, 
  Pytorch 1.11.0, 
  torchvision 0.12.0, 
  torchaudio 0.11.0
DATASETS
   ent_ids_1: Represents the entity IDs of the source knowledge graph.
   ent_ids_2: Represents the entity IDs of the target knowledge graph.
   triples_1: The relation quadruples (or temporal quadruples) encoded by IDs in the source knowledge graph.
   triples_2: The relation quadruples (or temporal quadruples) encoded by IDs in the target knowledge graph.
   rel_ids_1: Represents the relation IDs of the source knowledge graph.
   rel_ids_2: Represents the relation IDs of the target knowledge graph.
   sup_pairs + ref_pairs: The aligned entity pairs.
   ref_ent_ids: sup_pairs + ref_pairs.
   Note: Regarding the YAGO-WIKI50K and WIKI50K datasets, while the data format is based on Wikipedia + QID, we actually use the version with Wikipedia + entity names (i.e., the entity name version).

