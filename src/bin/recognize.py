#!/usr/bin/env python
import argparse
import json

import torch

import kaldi_io
from transformer import Transformer
from utils import add_results_to_json, process_dict
from data import build_LFR_features, load_video_frames

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Decoding.")
# data
parser.add_argument('--recog-json', type=str, required=True,
                    help='Filename of recognition data (json)')
parser.add_argument('--dict', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
parser.add_argument('--result-label', type=str, required=True,
                    help='Filename of result label data (json)')
# model
parser.add_argument('--model-path', type=str, required=True,
                    help='Path to model file created by training')
# decode
parser.add_argument('--beam-size', default=1, type=int,
                    help='Beam size')
parser.add_argument('--nbest', default=1, type=int,
                    help='Nbest size')
parser.add_argument('--decode-max-len', default=0, type=int,
                    help='Max output length. If ==0 (default), it uses a '
                    'end-detect function to automatically find maximum '
                    'hypothesis lengths')


def recognize(args):
    model, aud_LFR_m, aud_LFR_n, vid_LFR_m, vid_LFR_n = Transformer.load_model(args.model_path)
    print(model)
    model.eval()
    model.cuda()
    char_list, sos_id, eos_id = process_dict(args.dict)
    assert model.decoder.sos_id == sos_id and model.decoder.eos_id == eos_id

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    new_js = {}
    video_base_dir = "/mnt/data0/LipReadingData/LipReadingData_Proc/" # NOT BEST PRACTICE
    f = open("pcg_result.txt","a",encoding="utf-8")
    # s_ = ""
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            print('(%d/%d) decoding %s' %
                  (idx, len(js.keys()), name), flush=True)
            aud_input = kaldi_io.read_mat(js[name]['input'][0]['feat'])  # TxD
            vid_input = load_video_frames(video_base_dir, js[name]['utt2spk'], name)
            if (vid_input is None):
                continue
            if (aud_LFR_m != 1 or aud_LFR_n != 1):
                aud_input = build_LFR_features(aud_input, aud_LFR_m, aud_LFR_n)
            if (vid_LFR_m != 1 or vid_LFR_n != 1):
                vid_input = build_LFR_features(vid_input, vid_LFR_m, vid_LFR_n)

            aud_input = torch.from_numpy(aud_input).float()
            vid_input = torch.from_numpy(vid_input).float()
            aud_input_length = torch.tensor([aud_input.size(0)], dtype=torch.int)
            vid_input_length = torch.tensor([vid_input.size(0)], dtype=torch.int)
            aud_input = aud_input.cuda()
            vid_input = vid_input.cuda()
            aud_input_length = aud_input_length.cuda()
            vid_input_length = vid_input_length.cuda()

            nbest_hyps = model.recognize(aud_input, aud_input_length, vid_input, vid_input_length, char_list, args)
            new_js[name], gt_str, pred_str = add_results_to_json(js[name], nbest_hyps, char_list)
            f.write(str(idx) + "\n ground truth: " + gt_str + "\nprediction: " + pred_str + "\n\n")

    # f.write(s_)
    f.close()

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4,
                           sort_keys=True).encode('utf_8'))


if __name__ == "__main__":
    args = parser.parse_args()
    print(args, flush=True)
    recognize(args)
