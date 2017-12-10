exp={
	'test':{
		'learning_rate' : 1e-6,
		'training_iters' : 10000,
		'step_display' : 50,
		'step_save' : 500,
		'exp_name' : 'test',
		'pretrainedStep' : 0,
		'batch_size':40,
		'lam': 0.5,
		'joint_ratio':0.5,
		'plot':False,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"vgg"
	},

	'exp1':{
		'learning_rate' : 1e-4,
		'training_iters' : 20000,
		'step_display' : 50,
		'step_save' : 500,
		'exp_name' : 'exp1',
		'pretrainedStep' : 0,
		'batch_size':40,
		'lam': 0.5,
		'joint_ratio':0.5,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': True,
		'selectedmodel':"vgg_seg2"
	},

	'exp2':{
		'learning_rate' : 1e-5,
		'training_iters' : 20000,
		'step_display' : 50,
		'step_save' : 500,
		'exp_name' : 'exp2',
		'pretrainedStep' : 0,
		'batch_size':50,
		'lam': 0.5,
		'joint_ratio':0.5,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': True,
		'selectedmodel':"vgg_seg2"
	},

	'exp3':{
		'learning_rate' : 1e-4,
		'training_iters' : 20000,
		'step_display' : 50,
		'step_save' : 500,
		'exp_name' : 'exp3',
		'pretrainedStep' : 0,
		'batch_size':40,
		'lam': 0.5,
		'joint_ratio':0.5,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': True,
		'selectedmodel':"vgg_bn_seg2"
	},

	'exp4':{
		'learning_rate' : 1e-5,
		'training_iters' : 20000,
		'step_display' : 50,
		'step_save' : 500,
		'exp_name' : 'exp4',
		'pretrainedStep' : 10000,
		'batch_size':50,
		'lam': 0.5,
		'joint_ratio':0.5,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': True,
		'selectedmodel':"vgg_bn_seg2"
	},

	'exp401':{
		'learning_rate' : 1e-5,
		'training_iters' : 10000,
		'step_display' : 100,
		'step_save' : 500,
		'exp_name' : 'exp401',
		'pretrainedStep' : 10000,
		'batch_size':32,
		'lam': 0.5,
		'joint_ratio':1,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': True,
		'selectedmodel':"vgg_bn_seg2"
	},


	'expVal':{
		'learning_rate' : 0.001,
		'training_iters' : 10000,
		'step_display' : 100,
		'step_save' : 500,
		'exp_name' : 'expVal',
		'num' : '10000',
		'batch_size':64,
		'lam': 0.5,
		'plot':True,

		'train' : False,
		'validation' : True,
		'test': True,
		'selectedmodel':"vgg"
	}
}