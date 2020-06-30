# RGB-D Domain Adaptation
_Machine and Deep Learning Project @ Politecnico di Torino, Italy_

> The purpose, the achieved results and the theory behind these experiments are thoroughly explained in the [project report](#)

1. [Requirements](#requirements)
2. [Usage](#usage)
3. [References](#references)

## Requirements
Please check the requirements before running the experiments. You can find the necessary packages in the `requirement.txt` file.

## Usage
To run one of the experiments (between DA with relative rotation, Stepwise AFN and the combination of the two) just run

```bash
python3 main.py
```
The RR experiment will run with default parameters, if desired, provide options to the command according to the help description:

```
usage: main.py [-h] [--data_root DATA_ROOT] [--ram] [--no-ram]
               [--ckpt_dir CKPT_DIR] [--result_dir RESULT_DIR]
               [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
               [--class_num CLASS_NUM] [--dropout_p DROPOUT_P]
               [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
               [--step_size STEP_SIZE] [--gamma GAMMA] [--lambda LAMDA]
               [--entropy_weight ENTROPY_WEIGHT] [--dr DR] [--radius RADIUS]
               [--weight_L2norm WEIGHT_L2NORM]
               [--experiment {rr,safn,safn-rr}]

optional arguments:
  -h, --help            show this help message and exit
  --data_root DATA_ROOT
  --ram
  --no-ram
  --ckpt_dir CKPT_DIR   select --ckpt_dir=none if not desired
  --result_dir RESULT_DIR
  --batch_size BATCH_SIZE
  --epochs EPOCHS
  --lr LR
  --class_num CLASS_NUM
  --dropout_p DROPOUT_P
  --momentum MOMENTUM
  --weight_decay WEIGHT_DECAY
  --step_size STEP_SIZE
  --gamma GAMMA
  --lambda LAMDA        weight for pretext loss
  --entropy_weight ENTROPY_WEIGHT
                        weight for entropy loss
  --dr DR               step size for SAFN
  --weight_L2norm WEIGHT_L2NORM
                        weight of AFN loss
  --experiment {rr,safn,safn-rr}
                        select the experiment to run:`rr`: domain adaptation
                        with relative rotation,`safn`: stepwise afn,`safn-rr`:
                        stepwise afn and relative rotation DA
```

## References

[1] M.  R.  Loghmani,  L.  Robbiano,  M.  Planamente,  K.  Park,B.  Caputo,  and  M.  Vincze. _Unsupervised  domain  adaptation through inter-modal rotation for rgb-d object recognition,2020_
[2] R. Xu, G. Li, J. Yang, and L. Lin. _Larger norm more transferable: An adaptive feature norm approach for unsuperviseddomain adaptation, 2018_
