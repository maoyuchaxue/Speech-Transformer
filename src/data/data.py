"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the 
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.
"""
import json
import os
if __name__ == "__main__":
    import sys
    sys.path.append("../utils")


import numpy as np
import pickle
import torch
import torch.utils.data as data
from IPython import embed
import kaldi_io
from utils import IGNORE_ID, pad_list


class AudioVisualDataset(data.Dataset):
    """
    TODO: this is a little HACK now, put batch_size here now.
          remove batch_size to dataloader later.
    """

    def __init__(self, data_json_path, batch_size, max_length_in, max_length_out,
                 num_batches=0, batch_frames=0):
        # From: espnet/src/asr/asr_utils.py: make_batchset()
        """
        Args:
            data: espnet/espnet json format file.
            num_batches: for debug. only use num_batches minibatch but not all.
        """
        super(AudioVisualDataset, self).__init__()
        with open(data_json_path, 'rb') as f:
            data = json.load(f)['utts']
            # embed()
        # sort it by input lengths (long to short)
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['input'][0]['shape'][0]), reverse=True)
        # change batchsize depending on the input and output length
        minibatch = []
        # Method 1: Generate minibatch based on batch_size
        # i.e. each batch contains #batch_size utterances
        if batch_frames == 0:
            start = 0
            while True:
                ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
                olen = int(sorted_data[start][1]['output'][0]['shape'][0])
                factor = max(int(ilen / max_length_in), int(olen / max_length_out))
                # if ilen = 1000 and max_length_in = 800
                # then b = batchsize / 2
                # and max(1, .) avoids batchsize = 0
                b = max(1, int(batch_size / (1 + factor)))
                end = min(len(sorted_data), start + b)
                minibatch.append(sorted_data[start:end])
                # DEBUG
                # total= 0
                # for i in range(start, end):
                #     total += int(sorted_data[i][1]['input'][0]['shape'][0])
                # print(total, end-start)
                if end == len(sorted_data):
                    break
                start = end
        # Method 2: Generate minibatch based on batch_frames
        # i.e. each batch contains approximately #batch_frames frames
        else:  # batch_frames > 0
            print("NOTE: Generate minibatch based on batch_frames.")
            print("i.e. each batch contains approximately #batch_frames frames")
            start = 0
            while True:
                total_frames = 0
                end = start
                while total_frames < batch_frames and end < len(sorted_data):
                    ilen = int(sorted_data[end][1]['input'][0]['shape'][0])
                    total_frames += ilen
                    end += 1
                # print(total_frames, end-start)
                minibatch.append(sorted_data[start:end])
                if end == len(sorted_data):
                    break
                start = end
        if num_batches > 0:
            minibatch = minibatch[:num_batches]
        # embed()
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class AudioVisualDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, aud_LFR_m=1, aud_LFR_n=1, vid_LFR_m=1, vid_LFR_n=1, video_base_dir="/mnt/data0/LipReadingData/LipReadingData_Proc/", **kwargs):
        super(AudioVisualDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = LFRCollate(aud_LFR_m=aud_LFR_m, aud_LFR_n=aud_LFR_n, vid_LFR_m=vid_LFR_m, vid_LFR_n=vid_LFR_n, video_base_dir=video_base_dir)


class LFRCollate(object):
    """Build this wrapper to pass arguments(LFR_m, LFR_n) to _collate_fn"""
    def __init__(self, aud_LFR_m=1, aud_LFR_n=1, vid_LFR_m=1, vid_LFR_n=1, video_base_dir=""):
        self.aud_LFR_m = aud_LFR_m
        self.aud_LFR_n = aud_LFR_n
        self.vid_LFR_m = vid_LFR_m
        self.vid_LFR_n = vid_LFR_n
        self.video_base_dir = video_base_dir

    def __call__(self, batch):
        return _collate_fn(batch, aud_LFR_m=self.aud_LFR_m, aud_LFR_n=self.aud_LFR_n, vid_LFR_m=self.vid_LFR_m, vid_LFR_n=self.vid_LFR_n, video_base_dir=self.video_base_dir)


# From: espnet/src/asr/asr_pytorch.py: CustomConverter:__call__
def _collate_fn(batch, aud_LFR_m=1, aud_LFR_n=1, vid_LFR_m=1, vid_LFR_n=1, video_base_dir=""):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        xs_pad: N x Ti x D, torch.Tensor
        ilens : N, torch.Tentor
        ys_pad: N x To, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    batch = load_inputs_and_targets(batch[0], aud_LFR_m=aud_LFR_m, aud_LFR_n=aud_LFR_n, vid_LFR_m=vid_LFR_m, vid_LFR_n=vid_LFR_n, video_base_dir=video_base_dir)
    aud_xs, vid_xs, ys = batch

    # TODO: perform subsamping

    # get batch of lengths of input sequences
    aud_ilens = np.array([x.shape[0] for x in aud_xs])
    vid_ilens = np.array([x.shape[0] for x in vid_xs])

    if len(aud_xs) == 0 or len(vid_xs) == 0:
        return None

    # perform padding and convert to tensor
    aud_xs_pad = pad_list([torch.from_numpy(x).float() for x in aud_xs], 0)
    vid_xs_pad = pad_list([torch.from_numpy(x).float() for x in vid_xs], 0)
    aud_ilens = torch.from_numpy(aud_ilens)
    vid_ilens = torch.from_numpy(vid_ilens)
    ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], IGNORE_ID)
    return aud_xs_pad, aud_ilens, vid_xs_pad, vid_ilens, ys_pad


# ------------------------------ utils ------------------------------------
def load_inputs_and_targets(batch, aud_LFR_m=1, aud_LFR_n=1, vid_LFR_m=1, vid_LFR_n=1, video_base_dir=""):
    # From: espnet/src/asr/asr_utils.py: load_inputs_and_targets
    # load acoustic features and target sequence of token ids
    # for b in batch:
    #     print(b[1]['input'][0]['feat'])
    aud_xs = [kaldi_io.read_mat(b[1]['input'][0]['feat']) for b in batch]
    vid_xs = [load_video_frames(video_base_dir, b[1]['utt2spk'], b[0]) for b in batch]
    ys = [b[1]['output'][0]['tokenid'].split() for b in batch]

    if aud_LFR_m != 1 or aud_LFR_n != 1:
        # aud_xs = build_LFR_features(aud_xs, aud_LFR_m, aud_LFR_n)
        aud_xs = [build_LFR_features(x, aud_LFR_m, aud_LFR_n) for x in aud_xs]

    if vid_LFR_m != 1 or vid_LFR_n != 1:
        vid_xs = [build_LFR_features(x, vid_LFR_m, vid_LFR_n) if not x is None else None for x in vid_xs]

    # get index of non-zero length samples
    nonzero_idx = filter(lambda i: len(ys[i]) > 0 and not vid_xs[i] is None,
        range(len(aud_xs)))
    # sort in input lengths
    nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(aud_xs[i]))

    ### ?
    '''
    if len(nonzero_sorted_idx) != len(aud_xs):
        print("warning: Target sequences include empty tokenid")
        print(len(aud_xs), len(nonzero_sorted_idx))
    '''

    # remove zero-length samples
    aud_xs = [aud_xs[i] for i in nonzero_sorted_idx]
    vid_xs = [vid_xs[i] for i in nonzero_sorted_idx]
    ys = [np.fromiter(map(int, ys[i]), dtype=np.int64)
          for i in nonzero_sorted_idx]

    return aud_xs, vid_xs, ys

def load_video_frames(video_base_dir, speaker, utter_name):
    # returns video data in [nframes, channels, height, width]
    # normalized into [0, 1] range

    # AISHELL WORKAROUND, UGLY
    # When training mixed model, disable AISHELL
    
    if (speaker.startswith("S0")):
        return np.zeros((10, 3, 60, 80))
        #return None
    
    pkl_path = os.path.join(video_base_dir, speaker, utter_name + ".pkl")
    if (not os.path.exists(pkl_path) or os.path.getsize(pkl_path) < 100):
        # print(pkl_path)
        return None

    with open(pkl_path, "rb") as video_file:
        video_data = pickle.load(video_file)[0] / 255.0

        #apply normalization to fit pytorch pretrained model
        video_data = (video_data - 0.45) / 0.225

    return video_data


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.

    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.concatenate(inputs[i*n:i*n+m], axis=0))
        else: # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.concatenate(inputs[i*n:], axis=0)
            for _ in range(num_padding):
                frame = np.concatenate((frame, inputs[-1]), axis=0)
            LFR_inputs.append(frame)
    return np.stack(LFR_inputs, axis=0)
    #     LFR_inputs_batch.append(np.vstack(LFR_inputs))
    # return LFR_inputs_batch


if __name__ == "__main__":
    # dataloader & dataset test
    valid_json = "../../egs/aishell/dump/dev/deltafalse/data.json"
    cv_dataset = AudioVisualDataset(valid_json, 2,
                              800, 100)

    cv_loader = AudioVisualDataLoader(cv_dataset, batch_size=1,
                                num_workers=1,
                                aud_LFR_m=4, aud_LFR_n=3,
                                vid_LFR_m=3, vid_LFR_n=2)

    for i, (data) in enumerate(cv_loader):
        audio_padded_input, audio_input_lengths, video_padded_input, video_input_lengths, padded_target = data
        print(audio_padded_input.shape)
        print(audio_input_lengths.shape, audio_input_lengths)
        print(video_padded_input.shape)
        print(video_input_lengths.shape, video_input_lengths)
        print(padded_target.shape)
        print(audio_padded_input)
        print(video_padded_input)
        print(padded_target)
        break
