# from omegaconf import OmegaConf
# from rnn_trainer import BrainToTextDecoder_Trainer

# args = OmegaConf.load('rnn_args.yaml')
# trainer = BrainToTextDecoder_Trainer(args)
# metrics = trainer.train()

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
 
import warnings
warnings.filterwarnings("ignore")
 
 
if __name__ == "__main__":
    args = OmegaConf.load('rnn_args.yaml')
    trainer = BrainToTextDecoder_Trainer(args)
    metrics = trainer.train()