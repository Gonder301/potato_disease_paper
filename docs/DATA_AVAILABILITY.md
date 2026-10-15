# Data and code availability

The corrected v5 code, recorded predictions, epoch histories, frozen image manifest and partitions, analysis tables, hashes, reference metadata and figure materials are now distributed in this repository. The original final result archive is `results/results_closure_v5_final.zip`; its SHA256 is `501c4214e73583c628e799659b7e3a69b7aeaab38af28706e374c85ac3d71874`.

Image distributions are available at:

- https://www.kaggle.com/datasets/arjuntejaswi/plant-village
- https://www.kaggle.com/datasets/warcoder/potato-leaf-disease-dataset

Published leaf identity metadata is available at https://github.com/spMohanty/PlantVillage-Dataset/blob/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a/leaf-map.json.

The package contains 240 prediction files with 62,160 rows, 240 ten-epoch histories, 60 frozen partition files, the executed engine and continuation notebook, reanalysis code, four figures and eight illustrative source images. It does not include the complete image datasets or final trained weights. Pretrained-weight hashes and immutable download revisions were not recorded. The recorded predictions support metric reanalysis; future model training is not guaranteed to be bitwise identical.

Earlier July materials are preserved in `historical/july-2026/`. They do not define the corrected protocol or its conclusions. The old statement that this repository does not contain the corrected release has been superseded by this deposit. No data-package DOI has been assigned; the article DOI is a separate identifier.

## Suggested wording for production query Q12

After the corrected deposit is publicly accessible, the authors can replace the contradictory availability paragraph with the following text, using the stable repository version link listed in the README:

> The image datasets are publicly available at https://www.kaggle.com/datasets/arjuntejaswi/plant-village and https://www.kaggle.com/datasets/warcoder/potato-leaf-disease-dataset. Published leaf identity metadata is available at https://github.com/spMohanty/PlantVillage-Dataset/blob/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a/leaf-map.json. The corrected v5 reproducibility materials are publicly available at https://github.com/Gonder301/potato_disease_paper/tree/corrected-v5. They include frozen manifests and partitions, 240 per-image prediction files, epoch histories, hashes, the executed notebook engine, analysis scripts and figure materials. Eight illustrative source images are included; complete image datasets and final trained weights are not redistributed. The recorded predictions allow metric reanalysis, but future training is not guaranteed to be bitwise identical. Earlier materials are retained separately as historical versions. No DOI has been assigned to the reproducibility package.

This file documents the repository deposit. It does not submit a response to Frontiers or approve the proof. Author-name corrections and other production queries remain with the authors and journal.
