cd ~/code/robust_gnns_at_scale_publication/
conda activate rgnn_at_scale_refactoring




seml neurips21_cr_train add config/train/cora_and_citeseer.yaml
seml neurips21_cr_train add config/train/cora_and_citeseer_sgc.yaml
seml neurips21_cr_train add config/train/cora_and_citeseer_gat.yaml
seml neurips21_cr_train add config/train/cora_pprgo.yaml
seml neurips21_cr_train add config/train/citeseer_pprgo.yaml
seml neurips21_cr_train add config/train/cora_and_citeseer_rownorm.yaml
seml neurips21_cr_train add config/train/cora_and_citeseer_linear.yaml
seml neurips21_cr_train add config/train/cora_and_citeseer_direct.yaml

seml neurips21_cr_train add config/train/pubmed.yaml
seml neurips21_cr_train add config/train/arxiv.yaml
seml neurips21_cr_train add config/train/products.yaml
seml neurips21_cr_train add config/train/products_pprgo.yaml

seml neurips21_cr_train start



seml neurips21_cr_local_attack add config/attack_evasion_local_transfer/cora_and_citeseer_sga.yaml
seml neurips21_cr_local_attack add config/attack_evasion_local_transfer/cora_and_citeseer_prbcd.yaml
seml neurips21_cr_local_attack add config/attack_evasion_local_transfer/cora_and_citeseer_nettack.yaml

seml neurips21_cr_local_attack start