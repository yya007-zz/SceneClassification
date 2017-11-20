
exp2={
	'learning_rate' = 0.001
	'training_iters' = 50000
	'step_display' = 50
	'step_save' = 500
	'path_save' = './save/exp2/'
	'num = 500' #the model chosen to run on test data
	# start_from = './save/exp2/-'+str(num)
	'start_from' = ''

	'train' = True;
	'validation' = True;
	'selectedmodel'="VGG"
}

expVal={
	'learning_rate' = 0.001
	'training_iters' = 50000
	'step_display' = 50
	'step_save' = 500
	'path_save' = './save/exp2/'
	'num = 500' #the model chosen to run on test data
	# start_from = './save/exp2/-'+str(num)
	'start_from' = ''

	'train' = False;
	'validation' = True;
	'selectedmodel'="VGG"
}