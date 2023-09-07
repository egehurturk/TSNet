from imports import *
from models import *
from utils import *
from configuration import *

@torch.no_grad()
def test(args):
    model = DistNet.load_from_checkpoint(args['model'])
    model.eval()
    data = pickle.load(open(os.path.dirname(args['model']) + "/data.pickle", "rb"))

    x_test = data["x_test"]
    prev_sample = np.concatenate((np.zeros_like(x_test[0:1]), x_test[:-1]), axis=0)
    pad_x_test = np.concatenate((prev_sample, x_test), axis=2)

    y_pred = []
    for x in np.array_split(pad_x_test, 10):
        y_pred.append(model(torch.from_numpy(x)).numpy())

    y_pred = np.concatenate(y_pred)
    y_pred = y_pred[:, :, -x_test.shape[2] :]

    save(os.path.dirname(args['model']) + "/y_pred.wav", y_pred)
    save(os.path.dirname(args['model']) + "/x_test.wav", data["x_test"] * data["std"] + data["mean"])
    save(os.path.dirname(args['model']) + "/y_test.wav", data["y_test"])

def main():
    test_conf = asdict(TestConfiguration())
    test(test_conf)

if __name__ == "__main__":
    main()