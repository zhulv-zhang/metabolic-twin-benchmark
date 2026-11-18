# Data sources

Every external input the pipeline needs, with where to get it and where it lands.

| File | Source (URL) | Download date | Local landing path |
|---|---|---|---|
| TRRUST v2 mouse (trrust_rawdata.mouse.tsv) | https://www.grnpedia.org/trrust/ (download page) | <fill from file timestamp at build> | data/trrust_rawdata.mouse.tsv |
| GSE166336 supplement (mmc3.xlsx) | Matsuzaki et al., Cell Rep. 2021;36(8):109569, supplementary material (mirrored at GEO GSE166336) | <fill from file timestamp at build> | <local path read by prepare_real> |
| KEGG KGML (13 mouse pathways) | https://rest.kegg.jp/get/{pathway}/kgml | 30-31 August 2026 (re-verified 13 November 2026; per-batch timestamps in docs/KEGG_version_record.md) | kegg_kgml/ (created by step2_download_kgml.py) |
| KEGG API cache (list/mmu, link/*, get/rn) | https://rest.kegg.jp/ | same as above | GSE166336_小鼠组学数据/kegg_api_cache.json |

TRRUST is also shipped in data/ under CC BY-SA 4.0 (see LICENSES.md). KEGG raw
content is not shipped; it is downloaded at run time (see README).
