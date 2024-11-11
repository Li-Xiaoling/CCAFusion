
class args():

	# training args
	epochs = 10 #"number of training epochs"
	batch_size = 8 #"batch size for training"

	dataset_ir = "./data/ir" #" training dataset VTUAV(https://zhang-pengyu.github.io/DUT-VTUAV/)"
	dataset_vi = "./data/visible"  #" training dataset VTUAV(https://zhang-pengyu.github.io/DUT-VTUAV/)"


	HEIGHT = 256
	WIDTH = 256

	save_fusion_model = "./models/train/fusionnet/"
	save_loss_dir = './models/train/loss_fusionnet/'

	image_size = 256 #"size of training images, default is 256 X 256"
	# image_size = 256
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"

	lr = 1e-4 #"learning rate"
	log_interval = 10 #"number of images after which the training loss is logged"
	resume_fusion_model = None
	# nest net model
	resume_nestfuse: str = '/models/auto-encoder network.model'
	fusion_model = './models/train/fusionnet/'



