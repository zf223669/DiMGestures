import torch
import torch.nn.functional as F
import soundfile as sf
from fairseq import checkpoint_utils

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Model,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

model_path="/data/HOME/zhangfan/Chinese-speech-pretrain/hubert-base/chinese-hubert-base"
wav_path="/data/HOME/zhangfan/Chinese-speech-pretrain/wav/A1_1_m_xinwen_qingkuai.wav"
mask_prob=0.0
mask_length=10

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2Model.from_pretrained(model_path)

# for pretrain: Wav2Vec2ForPreTraining
# model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs Available: ", num_gpus)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model = model.half()
model.eval()

wav, sr = sf.read(wav_path)
input_values = feature_extractor(wav[:,0], return_tensors="pt",sr=sr).input_values
input_values = input_values.half()
input_values = input_values.to(device)

# for Wav2Vec2ForPreTraining
# batch_size, raw_sequence_length = input_values.shape
# sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
# mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.0, mask_length=2)
# mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)

with torch.no_grad():
    outputs = model(input_values)
    last_hidden_state = outputs.last_hidden_state
    print(last_hidden_state)
    print(f'shape: {last_hidden_state.shape}')
    # for Wav2Vec2ForPreTraining
    # outputs = model(input_values, mask_time_indices=mask_time_indices, output_hidden_states=True)
    # last_hidden_state = outputs.hidden_states[-1]