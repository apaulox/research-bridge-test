"""Mono + image reconstruction with one visual/audio contrastive feature pair."""
import torch


class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super().__init__()
        self.opt = opt
        self.net_visual, self.net_audio = nets

    def forward(self, input):
        mono = input['audio_mix_spec']
        visual, decoder_visual = self.net_visual(input['frame'])
        mask, audio = self.net_audio(mono, decoder_visual)
        real = mono[:, 0, :-1] * mask[:, 0] - mono[:, 1, :-1] * mask[:, 1]
        imag = mono[:, 0, :-1] * mask[:, 1] + mono[:, 1, :-1] * mask[:, 0]
        output = {
            'mask_prediction': mask,
            'binaural_spectrogram': torch.stack((real, imag), dim=1),
            'visual_feat': visual,
            'audio_feat': audio,
        }
        # Stereo ground truth is a loss target only, never an encoder input.
        if 'audio_diff_spec' in input:
            output['audio_gt'] = input['audio_diff_spec'][:, :, :-1].detach()
        return output
