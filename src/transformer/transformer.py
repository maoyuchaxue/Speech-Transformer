import torch
import torch.nn as nn

from decoder import Decoder
from video_encoder import VideoEncoder
from encoder import Encoder
from IPython import embed


class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, audio_encoder, video_encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = audio_encoder
        self.video_encoder = video_encoder
        self.decoder = decoder

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_audio_input, audio_input_lengths, padded_visual_input, visual_input_lengths, padded_target):
        """
        Args:
            padded_audio_input: N x Ti x D
            input_lengths: N
            padded_visual_input: N x Ti x 3 x H x W
            visual_input_lengths: N
            padded_targets: N x To
        """
        audio_encoder_padded_outputs, *_ = self.encoder(padded_audio_input, audio_input_lengths)
        visual_encoder_padded_outputs, *_ = self.video_encoder(padded_visual_input, visual_input_lengths)
        # pred is score before softmax
        pred, gold, *_ = self.decoder(padded_target, audio_encoder_padded_outputs, visual_encoder_padded_outputs, audio_input_lengths, visual_input_lengths)
        return pred, gold

    def recognize(self, audio_input, audio_input_length, visual_input, visual_input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input (audio & visual): T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        audio_encoder_outputs, *_ = self.encoder(audio_input.unsqueeze(0), audio_input_length)
        visual_encoder_outputs, *_ = self.video_encoder(visual_input.unsqueeze(0), visual_input_length)
        nbest_hyps = self.decoder.recognize_beam(audio_encoder_outputs[0], visual_encoder_outputs[0],
                                                 char_list,
                                                 args)
        return nbest_hyps

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model, aud_LFR_m, aud_LFR_n, vid_LFR_m, vid_LFR_n = cls.load_model_from_package(package)
        return model, aud_LFR_m, aud_LFR_n, vid_LFR_m, vid_LFR_n

    @classmethod
    def load_model_from_package(cls, package):
        print(package.keys())
        aud_encoder = Encoder(package['a_d_input'],
                          package['a_n_layers_enc'],
                          package['a_n_head'],
                          package['a_d_k'],
                          package['a_d_v'],
                          package['a_d_model'],
                          package['a_d_inner'],
                          dropout=package['a_dropout'],
                          pe_maxlen=package['a_pe_maxlen'])
        vis_encoder = VideoEncoder(package['v_d_input'],
                          package['v_n_layers_enc'],
                          package['v_n_head'],
                          package['v_d_k'],
                          package['v_d_v'],
                          package['v_d_model'],
                          package['v_d_inner'],
                          dropout=package['v_dropout'],
                          pe_maxlen=package['v_pe_maxlen'])
        
        decoder = Decoder(package['sos_id'],
                          package['eos_id'],
                          package['vocab_size'],
                          package['d_word_vec'],
                          package['n_layers_dec'],
                          package['a_n_head'],
                          package['a_d_k'],
                          package['a_d_v'],
                          package['a_d_model'],
                          package['a_d_inner'],
                          dropout=package['a_dropout'],
                          tgt_emb_prj_weight_sharing=package['tgt_emb_prj_weight_sharing'],
                          pe_maxlen=package['a_pe_maxlen'],
                          )
        model = cls(aud_encoder, vis_encoder, decoder)    
        print(model)
        model.load_state_dict(package['state_dict'])
        aud_LFR_m, aud_LFR_n, vid_LFR_m, vid_LFR_n = package['aud_LFR_m'], package['aud_LFR_n'], package['vid_LFR_m'], package['vid_LFR_n']
        return model, aud_LFR_m, aud_LFR_n, vid_LFR_m, vid_LFR_n

    @staticmethod
    def serialize(model, optimizer, epoch, aud_LFR_m, aud_LFR_n, vid_LFR_m, vid_LFR_n, tr_loss=None, cv_loss=None):
        package = {
            # Low Frame Rate Feature
            'aud_LFR_m': aud_LFR_m,
            'aud_LFR_n': aud_LFR_n,
            'vid_LFR_m': vid_LFR_m,
            'vid_LFR_n': vid_LFR_n,
            # audio encoder
            'a_d_input': model.encoder.d_input,
            'a_n_layers_enc': model.encoder.n_layers,
            'a_n_head': model.encoder.n_head,
            'a_d_k': model.encoder.d_k,
            'a_d_v': model.encoder.d_v,
            'a_d_model': model.encoder.d_model,
            'a_d_inner': model.encoder.d_inner,
            'a_dropout': model.encoder.dropout_rate,
            'a_pe_maxlen': model.encoder.pe_maxlen,
            # visual encoder
            'v_d_input': model.video_encoder.d_input,
            'v_n_layers_enc': model.video_encoder.n_layers,
            'v_n_head': model.video_encoder.n_head,
            'v_d_k': model.video_encoder.d_k,
            'v_d_v': model.video_encoder.d_v,
            'v_d_model': model.video_encoder.d_model,
            'v_d_inner': model.video_encoder.d_inner,
            'v_dropout': model.video_encoder.dropout_rate,
            'v_pe_maxlen': model.video_encoder.pe_maxlen,
            # decoder
            'sos_id': model.decoder.sos_id,
            'eos_id': model.decoder.eos_id,
            'vocab_size': model.decoder.n_tgt_vocab,
            'd_word_vec': model.decoder.d_word_vec,
            'n_layers_dec': model.decoder.n_layers,
            'tgt_emb_prj_weight_sharing': model.decoder.tgt_emb_prj_weight_sharing,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package
