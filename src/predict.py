from imports import *
from utils import *
from models import *
from configuration import *

@torch.no_grad()
def predict(args, train_configuration):
    model = DistNet(train_configuration)
    checkpoint = torch.load(args['model'])
    model.load_state_dict(checkpoint["state_dict"])

    # model = PedalNet.load_from_checkpoint(args['model'])
    model.eval()
    train_data = pickle.load(open(os.path.dirname(args['model']) + "/data.pickle", "rb"))

    mean, std = train_data["mean"], train_data["std"]

    # in_rate, in_data = wavfile.read(args.input)
    in_data, in_rate = librosa.load(args['input'], sr=44100)
    assert in_rate == 44100, "input data needs to be 44.1 kHz"
    sample_size = int(in_rate * args['sample_time'])
    length = len(in_data) - len(in_data) % sample_size

    # split into samples
    in_data = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    # standardize
    in_data = (in_data - mean) / std

    # pad each sample with previous sample
    prev_sample = np.concatenate((np.zeros_like(in_data[0:1]), in_data[:-1]), axis=0)
    pad_in_data = np.concatenate((prev_sample, in_data), axis=2)

    pred = []
    batches = math.ceil(pad_in_data.shape[0] / args['batch_size'])
    for x in tqdm(np.array_split(pad_in_data, batches)):
        pred.append(model(torch.from_numpy(x)).numpy())

    pred = np.concatenate(pred)
    pred = pred[:, :, -in_data.shape[2] :]

    save(args['output'], pred)

def main():
    p_conf = asdict(PredictionConfiguration('in_2.mp3', 'in_ts9_2.mp3'))
    train_conf = asdict(TrainConfiguration())
    predict(p_conf, train_conf)

if __name__ == "__main__":
    main()