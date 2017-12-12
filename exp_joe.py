exp_joe={
	'exp1':{
		'learning_rate_class' : 1e-5,
		'learning_rate_seg' : 1e-5,
		'training_iters' : 10000,
		'step_display' : 50,
		'step_save' : 500,
		'exp_name' : 'exp1',
		'pretrainedStep' : 0,
		'batch_size':16,
		'joint_ratio':1.0,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"vgg_seg1"
	},
	'exp2':{
		'learning_rate_class' : 5e-5,
		'learning_rate_seg' : 1e-5,
		'training_iters' : 10000,
		'step_display' : 50,
		'step_save' : 500,
		'exp_name' : 'exp2',
		'pretrainedStep' : 0,
		'batch_size':16,
		'joint_ratio':0.0,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"vgg_seg1"
	},
}
