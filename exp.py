exp={
	'test':{
		'learning_rate_class' : 1e-4,
		'learning_rate_seg' : 1e-5,
		'training_iters' : 200,
		'step_display' : 10,
		'step_save' : 500,
		'exp_name' : 'test',
		'pretrainedStep' : 0,
		'batch_size':32,
		'joint_ratio':1,
                'joint_ratio_decay': False,
		'plot':True,
		'lr_decay':True,

		'train' : True,
		'validation' : False,
		'test': False,
		'selectedmodel':"vgg_bn_seg2"
	},

	'exp1':{
		'learning_rate_class' : 1e-4,
		'learning_rate_seg' : 1e-4,
		'training_iters' : 20000,
		'step_display' : 100,
		'step_save' : 1000,
		'exp_name' : 'exp1',
		'pretrainedStep' : 0,
		'batch_size':32,
		'joint_ratio':0.5,
                'joint_ratio_decay': False,
		'plot':True,
		'lr_decay':True,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"vgg_bn_seg2"
	},

	'exp2':{
		'learning_rate_class' : 1e-4,
		'learning_rate_seg' : 1e-4,
		'training_iters' : 20000,
		'step_display' : 100,
		'step_save' : 1000,
		'exp_name' : 'exp2',
		'pretrainedStep' : 0,
		'batch_size':32,
		'joint_ratio':0.5,
                'joint_ratio_decay': False,
		'plot':True,
		'lr_decay':True,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"vgg_bn_seg2_1"
	},

	'exp3':{
		'learning_rate_class' : 1e-4,
		'learning_rate_seg' : 1e-4,
		'training_iters' : 20000,
		'step_display' : 100,
		'step_save' : 1000,
		'exp_name' : 'exp3',
		'pretrainedStep' : 0,
		'batch_size':32,
		'joint_ratio':0.1,
                'joint_ratio_decay': False,
		'plot':True,
		'lr_decay':True,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"vgg_bn_seg2_1"
	},

	'exp4':{
		'learning_rate_class' : 1e-4,
		'learning_rate_seg' : 1e-4,
		'training_iters' : 20000,
		'step_display' : 100,
		'step_save' : 1000,
		'exp_name' : 'exp4',
		'pretrainedStep' : 0,
		'batch_size':32,
		'joint_ratio':0.5,
                'joint_ratio_decay': False,
		'plot':True,
		'lr_decay':True,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"vgg_bn_seg2_2"
	},

	'exp5':{
		'learning_rate_class' : 1e-4,
		'learning_rate_seg' : 1e-4,
		'training_iters' : 20000,
		'step_display' : 100,
		'step_save' : 1000,
		'exp_name' : 'exp5',
		'pretrainedStep' : 0,
		'batch_size':32,
		'joint_ratio':0.1,
                'joint_ratio_decay': False,
		'plot':True,

		'train' : True,
		'validation' : True,
		'test': False,
		'selectedmodel':"vgg_bn_seg2_2"
	},
	'exp6':{
		'learning_rate_class' : 1e-4,
		'learning_rate_seg' : 1e-5,
		'training_iters' : 20000,
		'step_display' : 50,
		'step_save' : 500,
		'exp_name' : 'exp2',
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
