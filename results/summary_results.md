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

# RGBD_DA_RR

	{
		'gamma': 0.02, 
		'lr': 0.00014563484775012445, 
		'step_size': 3
	} 
	
	accuracy: 0.5813215913289814 
	epoch: 6

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
	
## depth_only
	
	{
		'gamma': 0.05,
		'lr': 0.005689866029018299,
		'step_size': 5
	}
	
	accuracy: 0.2709693311984234 
	epoch: 6

	
## e2e

	{
		'gamma': 0.3, 
		'lr': 0.0005179474679231213,
		'step_size': 3
	} 
	
	accuracy: 0.5718068727675822 
	epoch: 2


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
	
# only_HAFN

- Batch size: 32
- radius: 25

## RGB_only
	
	
	
## depth_only
	
	

	
## e2e

	


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


