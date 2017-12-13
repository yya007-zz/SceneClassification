exp={
	'vgg_objnet':{
		'learning_rate_class' : 1e-4,
		'learning_rate_seg' : 1e-4,
		'training_iters' : 20000,
		'step_display' : 100,
		'step_save' : 1000,
		'exp_name' : 'vgg_objnet',
		'pretrainedStep' : 0,
		'batch_size':32,
		'joint_ratio':0.1,
                'joint_ratio_decay': False,
		'plot':True,
                'lr_decay': True,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"vgg_objnet"
	},
	'vgg_segnet':{
		'learning_rate_class' : 1e-4,
		'learning_rate_seg' : 1e-5,
		'training_iters' : 20000,
		'step_display' : 50,
		'step_save' : 500,
		'exp_name' : 'vgg_segnet',
		'pretrainedStep' : 0,
		'batch_size':14,
		'joint_ratio':0.1,
                'joint_ratio_decay': True,
		'plot':True,
                'lr_decay': True,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"vgg_segnet"
	},

	'baseline':{
		'learning_rate_class' : 1e-3,
		'learning_rate_seg' : 1e-3,
		'training_iters' : 40000,
		'step_display' : 100,
		'step_save' : 5000,
		'exp_name' : 'baseline',
		'pretrainedStep' : 0,
		'batch_size':256,
		'joint_ratio':0,
                'joint_ratio_decay': False,
		'plot':True,
		'lr_decay':False,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"alexnet"
	}
}
