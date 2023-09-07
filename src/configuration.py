from dataclasses import dataclass, astuple, asdict

MODEL = "models/distnet/distnet.ckpt"

@dataclass
class TrainConfiguration:
  """
  Training configurations:
  * in_file: the path to input file with undistorted, clean signal (i.e., x)
  * out_file: the path to the output file with distorted and TS9 applied signal (i.e., f(x))
  * sample_time
  * normalize: normalize the signals in the [0, 1] range to make it stable in training
  * num_channels: the number of channels in WaveNet
  * dilation_depth: the depth of dilation applied in WaveNet
  * num_repeat
  * kernel_size: the shape of the kernel used in convolution
  * batch_size
  * learning_Rate
  * max_epochs
  * gpus
  * tpu_cores
  * cpu
  * model: the path to the saved model
  * resume
  """
  in_file: str = "in_1.mp3"
  out_file: str = "in_ts9_1.mp3"
  sample_time: float = 100e-3
  normalize: bool = True
  num_channels: int = 4
  dilation_depth: int = 9
  num_repeat: int = 2
  kernel_size: int = 3
  batch_size: int = 64
  learning_rate: float = 3e-3
  max_epochs: int = 1000
  gpus: int = -1
  tpu_cores: bool = None
  cpu: bool = True
  model: str = MODEL
  resume: bool = False


@dataclass
class TestConfiguration:
  """
  Test configurations
  * model: the path to model
  """
  model: str = MODEL

@dataclass
class PredictionConfiguration:
  """
  Prediction configurations
  * input: the input, clean signal to predict
  * output: the distorted signal, the truth
  * model: path to model
  * batch_size
  * sample_time
  """
  input: str
  output: str
  model:str = MODEL
  batch_size: int = 256
  sample_time: float = 100e-3

@dataclass
class PlotConfiguration:
  """
  Plotting configurations
  * model: path to model
  * show: show the plot in notebook
  """
  model: str = MODEL
  show: bool = True

