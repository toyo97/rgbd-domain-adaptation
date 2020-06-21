NOTE: epoch goes from 0 on

# BASELINE

## RGB_only

	{
		'gamma': 0.05,
		'lr': 6.250551925273976e-05,
		'step_size': 7
	}

	accuracy: 0.5196760684813401
	epoch: 8

## depth_only

	{
		'gamma': 0.05,
		'lr': 0.008685113737513529,
		'step_size': 5
	}

	accuracy: 0.15445251878310137
	epoch: 5

## e2e

	{
		'gamma': 0.1,
		'lr': 0.00029470517025518097,
		'step_size': 2
	}
	
	accuracy: 0.4734265303608819
	epoch: 7

# RGBD_DA_RR

## weight_decay

	Best configuration

	'weight_decay': 0.0005
		
	accuracy: 0.5979800468037936
	epoch: 9
	
	
# only_SAFN

## RGB_only

### tuning

 	{
		'gamma': 0.02,
		'lr': 0.0007906043210907702,
		'step_size': 7
	}
	
	accuracy: 0.5830151496489715 		epoch: 1
	
### default

	{
		'dr': 1,
		'weight_decay': 0.05,
		'lr': 0.0003,
		'entropy_weight': 0.1,
		'weight_l2norm': 0.05,
		'batch_size': 32
	}
	
	accuracy: 0.6222749107032886
	epoch: 1

	
## depth_only

### tuning

	{
		'gamma': 0.1,
		'lr': 0.0037275937203149418,
		'step_size': 7
	}
	
	accuracy: 0.3084739499938416 		epoch: 0

### default

	{
		'dr': 1,
		'weight_decay': 0.05,
		'lr': 0.0003,
		'entropy_weight': 0.1,
		'weight_l2norm': 0.05,
		'batch_size': 32
	}
	
	accuracy: 0.26210124399556595
	epoch: 1
	
## e2e

### tuning

	{
		'gamma': 0.3,
		'lr': 0.0024420530945486497,
		'step_size': 7
	}
	
	accuracy: 0.5762717083384653
	epoch: 0
	
### default

	{
		'dr': 1,
		'weight_decay': 0.05,
		'lr': 0.0003,
		'entropy_weight': 0.1,
		'weight_l2norm': 0.05,
		'batch_size': 32
	}
	
	accuracy: 0.5213696268013303
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
	epoch: 9


