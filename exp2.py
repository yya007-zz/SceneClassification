exp2={
	'exp1':{
		'learning_rate' : 1e-5,
		'training_iters' : 10000,
		'step_display' : 100,
		'step_save' : 500,
		'exp_name' : 'exp1',
		'pretrainedStep' : 0,
		'batch_size':16,
		'lam': 0.5,
		'joint_ratio':1,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': True,
		'selectedmodel':"vgg_seg1"
	},
	'exp2':{
		'learning_rate' : 1e-5,
		'training_iters' : 10000,
		'step_display' : 100,
		'step_save' : 500,
		'exp_name' : 'exp2',
		'pretrainedStep' : 0,
		'batch_size':16,
		'lam': 0.1,
		'joint_ratio':0.5,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': True,
		'selectedmodel':"vgg_seg1"
	},
	'exp3':{
		'learning_rate' : 1e-5,
		'training_iters' : 10000,
		'step_display' : 100,
		'step_save' : 500,
		'exp_name' : 'exp3',
		'pretrainedStep' : 0,
		'batch_size':16,
		'lam': 0.1,
		'joint_ratio':0.2,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': True,
		'selectedmodel':"vgg_seg1"
	},
}
