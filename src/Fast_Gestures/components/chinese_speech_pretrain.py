import torch
from torch import nn
import torch.nn.functional as F
import soundfile as sf
# from fairseq import checkpoint_utils
from src import utils

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)
from os import path

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Model,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
# model_path = "/home/lmj2/PG/src/utils/chinese_speech_pretrain/hubert-large/chinese-hubert-large"
model_path = "/home/lmj2/PG/src/utils/chinese_speech_pretrain/hubert-base/chinese-hubert-base"
wav_path = "/data/HOME/zhangfan/Chinese_Neutral-speech-pretrain/wav/A1_1_m_xinwen_qingkuai.wav"
mask_prob = 0.0
mask_length = 10


class HubertBase(nn.Module):
    def __init__(self,
                 checkpoint_path='./hubert-base/chinese-hubert-base',
                 model_path = './hubert-base/chinese-hubert-base'):
        super().__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_path)
        self.model = Wav2Vec2Model.from_pretrained(model_path)
        log.info(f'checkpoint_path: {checkpoint_path}')
        log.info(f'model_path: {model_path}')
        # self.model = self.model.to(self.device)
        # self.model = self.model.half()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, audio_dataset):
        # input_values = self.feature_extractor(audio_dataset, return_tensors="pt", sampling_rate=16000).input_values[0]
        # input_values = input_values.to(self.device)
        # input_values = input_values.half()
        left_pad = int(16000 * 0.2)
        right_pad = int(16000 * 0.2)
        with torch.no_grad():
            audio_dataset = F.pad(audio_dataset, (left_pad, right_pad), "constant", 0)
            outputs = self.model(audio_dataset)
            # last_hidden_state = outputs.last_hidden_state[:, :999]
            # last_hidden_state = outputs.last_hidden_state[:, :999]
            # print(last_hidden_state)
            # print(f'shape: {last_hidden_state.shape}')
            return 'Chinese', outputs.last_hidden_state[:, :999]
