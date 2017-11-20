exp={
	'exp1':{
		'learning_rate' : 0.0001,
		'training_iters' : 10000,
		'step_display' : 100,
		'step_save' : 500,
		'exp_name' : 'exp1',
		'num' : '6000',
		'batch_size':64,

		'train' : True,
		'validation' : True,
		'selectedmodel':"VGG"
	},

	'exp2':{
		'learning_rate' : 0.0001,
		'training_iters' : 10000,
		'step_display' : 100,
		'step_save' : 500,
		'exp_name' : 'exp2',
		'num' : '',
		'batch_size':40,

		'train' : True,
		'validation' : True,
		'selectedmodel':"VGG_BN"
	},

	'exp3':{
		'learning_rate' : 0.01,
		'training_iters' : 10000,
		'step_display' : 100,
		'step_save' : 500,
		'exp_name' : 'exp3',
		'num' : '',
		'batch_size':40

		'train' : True,
		'validation' : True,
		'selectedmodel':"VGG_BN"
	},


	'expVal':{
		'learning_rate' : 0.001,
		'training_iters' : 10000,
		'step_display' : 100,
		'step_save' : 500,
		'exp_name' : 'exp1',
		'num' : '10000',
		'batch_size':64

		'train' : False,
		'validation' : True,
		'selectedmodel':"VGG"
	}
}