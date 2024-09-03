## Released Models


| Model Name | Link | MD5 Hash | Matbench Discovery | D3 Corrections | Notes |
|------------|------|----------|---------------------|-----------------|-------|
| **orb-v1** | [link](https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orbff-v1-20240827.ckpt) | 92897eda08609425ee001955c7885139 | Yes | No | Full dataset pretraining, MPTraj + Alexandria finetuning |
| **orb-mptraj-only-v1** | [link](https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orbff-mptraj-only-v1-20240827.ckpt) | ff42a2bc1e1f50b5f3ee2a20b83cf3a2 | Yes | No | MPTraj pretraining and finetuning only |
| **orb-d3-v1** | [link](https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orb-d3-v1-20240902.ckpt) | 470c7d3482ead3bc97cd4b46382d5e47 | No | Yes | Full dataset pretraining, MPTraj + Alexandria finetuning, integrated D3 corrections |
| **orb-d3-sm-v1** | [link](https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orb-d3-sm-v1-20240902.ckpt) | 64fe91603e46ad5fa695525e3f1e9397 | No | Yes | First 10 layers of a pretrained model finetuned on mptrj + alexandria with D3 corrections |
| **orb-d3-xs-v1** | [link](https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orb-d3-xs-v1-20240902.ckpt) | 79d042f9f16c4407795426a75498fbb7 | No | Yes | First 5 layers of a pretrained model finetuned on mptrj + alexandria with D3 corrections |



### Matbench Discovery Results


### orb-v1: Full dataset pretraining, MPtraj + Alexandria finetuning

```
                      orb          10k         unique
F1              0.846577     0.988213       0.867282
DAF             5.394101     6.389021       6.015771
Precision       0.898971     0.976700       0.919641
Recall          0.799953     1.000000       0.820563
Accuracy        0.951678     0.976700       0.961608
TPR             0.799953     1.000000       0.820563
FPR             0.017979     1.000000       0.012939
TNR             0.982021     0.000000       0.987061
FNR             0.200047     0.000000       0.179437
TP          34258.000000  9767.000000   27031.000000
FP           3850.000000   233.000000    2362.000000
TN         210288.000000     0.000000  180184.000000
FN           8567.000000     0.000000    5911.000000
MAE             0.030884     0.019012       0.030589
RMSE            0.080986     0.064470       0.079003
R2              0.798803     0.907903       0.815941
```
### orb-mptraj-only-v1: MPTraj pretraining, MPTraj finetuning

```
                     orb          10k         unique
F1              0.752143     0.963193       0.761336
DAF             4.267540     6.076994       4.667345
Precision       0.711221     0.929000       0.713505
Recall          0.798062     1.000000       0.816040
Accuracy        0.912341     0.929000       0.921787
TPR             0.798062     1.000000       0.816040
FPR             0.064804     1.000000       0.059130
TNR             0.935196     0.000000       0.940870
FNR             0.201938     0.000000       0.183960
TP          34177.000000  9290.000000   26882.000000
FP          13877.000000   710.000000   10794.000000
TN         200261.000000     0.000000  171752.000000
FN           8648.000000     0.000000    6060.000000
MAE             0.044745     0.040998       0.046230
RMSE            0.093426     0.102950       0.093919
R2              0.732243     0.780546       0.739879
```

