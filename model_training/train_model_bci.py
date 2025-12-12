from omegaconf import OmegaConf
from FastBCI import BrainToTextDecoder_Trainer

args = OmegaConf.load('rnn_args2.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()
