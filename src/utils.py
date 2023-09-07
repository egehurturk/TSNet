from imports import *
# helpful functions

def error_to_signal(y, y_pred, use_filter=1):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    if use_filter == 1:
        y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return np.sum(np.power(y - y_pred, 2)) / (np.sum(np.power(y, 2) + 1e-10))


def pre_emphasis_filter(x, coeff=0.95):
    return np.concatenate([x, np.subtract(x, np.multiply(x, coeff))])

def read_wave(wav_file):
    # Extract Audio and framerate from Wav File
    fs, signal = wavfile.read(wav_file)
    return signal, fs

def equalize_lengths(in_data, out_data):
    if len(in_data) > len(out_data):
      print("Trimming input audio to match output audio")
      in_data = in_data[0:len(out_data)]
    if len(out_data) > len(in_data):
      print("Trimming output audio to match input audio")
      out_data = out_data[0:len(in_data)]
    return in_data, out_data

def process_stereo(in_data, out_data):
    #If stereo data, use channel 0
    if len(in_data.shape) > 1:
        print("[WARNING] Stereo data detected for in_data, only using first channel (left channel)")
        in_data = in_data[:,0]
    if len(out_data.shape) > 1:
        print("[WARNING] Stereo data detected for out_data, only using first channel (left channel)")
        out_data = out_data[:,0]
    return in_data, out_data

def convert_fp32(in_data, out_data):
    # Convert PCM16 to FP32
    if in_data.dtype == "int16":
        in_data = in_data/32767
        print("In data converted from PCM16 to FP32")
    if out_data.dtype == "int16":
        out_data = out_data/32767
        print("Out data converted from PCM16 to FP32")
    return in_data, out_data

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm

def save(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def prepare(args):
    in_data, in_rate = librosa.load(args['in_file'], sr=44100)
    out_data, out_rate = librosa.load(args['out_file'], sr=44100)
    assert in_rate == out_rate, "in_file and out_file must have same sample rate"

    # Trim the length of audio to equal the smaller wav file
    in_data, out_data = equalize_lengths(in_data, out_data)

    in_data, out_data = process_stereo(in_data, out_data)

    in_data, out_data = convert_fp32(in_data, out_data)

    #normalize data
    if args['normalize'] == True:
        in_data = normalize(in_data)
        out_data = normalize(out_data)

    sample_size = int(in_rate * args['sample_time'])
    length = len(in_data) - len(in_data) % sample_size

    x = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)
    y = out_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    split = lambda d: np.split(d, [int(len(d) * 0.6), int(len(d) * 0.8)])

    d = {}
    d["x_train"], d["x_valid"], d["x_test"] = split(x)
    d["y_train"], d["y_valid"], d["y_test"] = split(y)
    d["mean"], d["std"] = d["x_train"].mean(), d["x_train"].std()

    # standardize
    for key in "x_train", "x_valid", "x_test":
        d[key] = (d[key] - d["mean"]) / d["std"]

    if not os.path.exists(os.path.dirname(args['model'])):
        os.makedirs(os.path.dirname(args['model']))

    pickle.dump(d, open(os.path.dirname(args['model']) + "/data.pickle", "wb"))

def props(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            pr[name] = value
    return pr