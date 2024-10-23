# import os
# from melo.api import TTS
# import torch
# from openvoice import se_extractor
# from openvoice.api import ToneColorConverter

# texts = {
#     'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
#     'EN': "Did you ever hear a folk tale about a giant turtle?",
# }
# output_dir = 'outputs_v1'
# os.makedirs(output_dir, exist_ok=True)

# src_path = f'{output_dir}/tmp.wav'

# # Speed is adjustable
# speed = 1.0
# device="cpu"
# for language, text in texts.items():
#     model = TTS(language=language, device=device)
#     speaker_ids = model.hps.data.spk2id
    
#     for speaker_key in speaker_ids.keys():
#         speaker_id = speaker_ids[speaker_key]
#         speaker_key = speaker_key.lower().replace('_', '-')
        
#         source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
#         model.tts_to_file(text, speaker_id, src_path, speed=speed)
#         save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

#         # Run the tone color converter
#         encode_message = "@MyShell"
#         tone_color_converter.convert(
#             audio_src_path=src_path, 
#             src_se=source_se, 
#             tgt_se=target_se, 
#             output_path=save_path,
#             message=encode_message)


import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device="cpu"
output_dir = 'outputs'

base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)




