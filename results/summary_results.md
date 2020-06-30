# Configurations used to run the final 5 experiments to obtain the average results.

Here are reported the hyperparameters which differ from the default setting written on the paper:
- Learning rate: 0.0003
- Learning rate fixed during all the training process
- Weight for the loss of the Pretext Head (lambda_p): 1.0
- Entropy Loss Weight (lambda_h): 0.1
- Batch size: 64
- Stochastic Gradient Descent Optimizer 
- Momentum: 0.9
- Weight decay: 0.05
- Dropout: 0.5


# BASELINE

## RGB_only

	{
		'gamma': 0.05,
		'lr': 6.250551925273976e-05,
		'step_size': 7
	}

	accuracy: 0.5196760684813401
	epoch: 9
	
	After 5 runs
	Average result obtained: 0.5058627909841114
	standard deviation: 0.016668470952876515

## depth_only

	{
		'gamma': 0.05,
		'lr': 0.008685113737513529,
		'step_size': 5
	}

	accuracy: 0.15445251878310137
	epoch: 6
	
	After 5 runs
	Average result obtained: 0.18647616701564232
	standard deviation: 0.007759087525686346

## e2e

	{
		'gamma': 0.1,
		'lr': 0.00029470517025518097,
		'step_size': 2
	}
	
	accuracy: 0.4734265303608819
	epoch: 8
	
	After 5 runs
	Average result obtained: 0.43235620150264814 
	standard deviation: 0.012487841043316773

# RGBD_DA_RR

	{
		'gamma': 0.02, 
		'lr': 0.00014563484775012445, 
		'step_size': 3
	} 
	
	accuracy: 0.5813215913289814 
	epoch: 6
	
	After 2 run
	Average result obtained: 0.5764860204458677
	standard deviation: 0.0033304594161842616

## weight_decay 0.0005

	{
		'gamma': 0.02, 
		'lr': 0.0004498432668969444, 
		'step_size': 7
	} 
	
	accuracy: 0.5142258898879172
	epoch: 9
		
# only_SAFN

- Batch size: 32
- delta_r: 1.0
- weight L2 norm: 0.05

## RGB_only
	
	Default hyperparameters
	
	{
		'dr': 1,
		'weight_decay': 0.05, 
		'lr': 0.0003, 
		'entropy_weight': 0.1, 
		'weight_l2norm': 0.05, 
		'batch_size': 32
	}
	
	accuracy: 0.6222749107032886 
	epoch: 2
	
	After 5 runs
	Average result obtained: 0.5794063308289198 
	standard deviation: 0.010835648112320767
	
	Results with Normal Dropout
	accuracy: 0.5751016134991994
	epoch: 1
	

## depth_only
	
	{
		'gamma': 0.05,
		'lr': 0.005689866029018299,
		'step_size': 5
	}
	
	accuracy: 0.2709693311984234 
	epoch: 6
	
	After 5 runs
	Average result obtained: 0.2678716590713142
	standard deviation: 0.01339667400881486

	Results with Normal Dropout
	accuracy: 0.1272940017243503
	epoch: 1
	
	
## e2e

	{
		'gamma': 0.3, 
		'lr': 0.0005179474679231213,
		'step_size': 3
	} 
	
	accuracy: 0.5718068727675822 
	epoch: 2
	
	After 5 runs
	Average result obtained: 0.5565402143121074
	standard deviation: 0.02125088893518903
	
	Results with Normal Dropout
	accuracy: 0.5845547481216898
	epoch: 1


# SAFN_RR

	{
		'entropy_weight': 0.05,
		'lamda': 0.8,
		'lr': 5.4286754393238594e-05, 
		'weight_L2norm': 0.05,
		'weight_decay': 0.0005
	} 
	
	accuracy: 0.5624461140534549 
	epoch: 10
	
	After 2 runs
	Average result obtained: 0.5518844685306072
	standard deviation: 0.02013794802315555
	
## Ablation studies
### Pretext Head = Main Head

accuracy: 0.5571191033378495
epoch: 9

### No entropy loss

accuracy: 0.5278051484172928
epoch: 10

### No source dataset through pretext task

accuracy: 0.5444636038921049
epoch: 10

### No factor after dropout

accuracy: 0.5339019583692572
epoch: 10
	
# only_HAFN

- Batch size: 32
- radius: 25
- weight L2 norm: 0.05

## RGB_only
	
	{
		'gamma': 0.02, 
		'lr': 0.0003906939937054617,
		'step_size': 3
	} 
	accuracy: 0.6019830028328612 
	epoch: 1
	
	After 5 run
	Average result obtained: 0.5697622859958122 
	standard deviation: 0.006180266613741486
	
	
## depth_only
	
	{
	'gamma': 0.05, 
	'lr': 0.007543120063354615, 
	'step_size': 6
	} 
	accuracy: 0.2971732972040892 
	epoch: 8
	
	After 5 run
	Average result obtained: 0.28486266781623354
	standard deviation: 0.010440746376055843
		
	
## e2e

	{
		'gamma': 0.05, 
		'lr': 0.0005179474679231213, 
		'step_size': 2
	} 
	accuracy: 0.5866486020445868
	epoch: 3
	
	After 5 run
	Average result obtained: 0.5539967976351768 
	standard deviation: 0.023607242092620582
	

# HAFN_RR

	{
	'entropy_weight': 0.05,
	'lamda': 0.5, 
	'lr': 7.196856730011529e-05,
	'weight_L2norm': 0.05, 
	'weight_decay': 0.05
	} 
	accuracy: 0.5723303362483064 
	epoch: 9
	
	After 2 runs
	Average result obtained: 0.5514841729277005 
	standard deviation: 0.004803547234881134
	
