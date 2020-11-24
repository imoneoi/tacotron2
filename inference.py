import torch

from hparams import create_hparams
from model import Tacotron2
from train import load_model
from text import text_to_sequence
from text.pinyin_processing import process_line_data, get_autolabeled_prosody, process_autolabeled_text

import soundfile
import numpy as np

import argparse
import os

from tqdm import tqdm

# include WaveGlow
import sys
sys.path.append("../waveglow")
from denoiser import Denoiser

MAX_WAV_VALUE = 32767.0

def load_task_list(tasklist_path, out_path, task_repeat=3):
    f = open(tasklist_path, "r")
    lines = f.readlines()
    f.close()

    result = []
    for line in lines:
        text, filename = line.rstrip("\n").split("|")

        for i in range(task_repeat):
            result.append([text, os.path.join(out_path, filename + " (" + str(i) + ").wav")])

    return result

def inference(
    task_list,

    taco_checkpoint,
    taco_hparams,
    
    waveglow_checkpoint,
    waveglow_sampling_rate=22050,
    waveglow_sigma=0.6,
    waveglow_denoiser_strength=0.1
):
    # Load tacotron
    taco_model = load_model(taco_hparams)
    taco_model.load_state_dict(torch.load(taco_checkpoint)['state_dict'])

    # Evaluation mode (no Dropout)
    taco_model.eval().cuda()
    assert taco_model.training == False

    # Load waveglow
    waveglow = torch.load(waveglow_checkpoint)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)

    # Evaluation mode
    waveglow.eval().cuda()
    assert waveglow.training == False

    # Denoiser for waveglow
    if waveglow_denoiser_strength > 0:
        denoiser = Denoiser(waveglow)

    # Inference
    for task in tqdm(task_list):
        raw_text, out_file = task

        # Split-Concat mechanism
        final_audio = []
        raw_text_splitted = raw_text.split("/")

        for raw_text_part in raw_text_splitted:
            # Get input text
            prosody = get_autolabeled_prosody(raw_text_part)
            text = process_autolabeled_text(raw_text_part)
            input_text = process_line_data(text, prosody)

            # Get input sequence
            input_seq = torch.tensor(np.array(text_to_sequence(input_text, [])), dtype=torch.long).cuda()
            input_seq = input_seq.unsqueeze(0)

            # Inference
            with torch.no_grad():
                # Tacotron 2
                mel_outputs, mel_outputs_postnet, _, alignments = taco_model.inference(input_seq)

                # Waveglow
                audio = waveglow.infer(mel_outputs_postnet, sigma=waveglow_sigma)
                if waveglow_denoiser_strength > 0:
                    audio = denoiser(audio, waveglow_denoiser_strength)

                # to CPU
                audio = audio.squeeze()
                audio = audio.cpu().numpy()

            # Concat
            final_audio = np.concatenate([final_audio, audio])

        # Write output audio
        final_audio = (final_audio * MAX_WAV_VALUE).astype(np.int16) # cast to 16bit audio
        soundfile.write(out_file, final_audio, waveglow_sampling_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_list", type=str, default="inference_text.txt", help="Path to task list file")
    parser.add_argument("--out_path", type=str, default="inference_result", help="Path to result")

    parser.add_argument("--taco_checkpoint", type=str, default="outdir_remote/checkpoint_19000", help="Tacotron 2 checkpoint")
    parser.add_argument("--waveglow_checkpoint", type=str, default="../waveglow/checkpoints/waveglow_24000", help="Tacotron 2 checkpoint")

    args = parser.parse_args()

    # Taco hparams
    taco_hparams = create_hparams()
    taco_hparams.distributed_run = False
    taco_hparams.max_decoder_steps = 4096

    # Load task list
    task_list = load_task_list(args.task_list, args.out_path)

    # Inference
    inference(task_list, args.taco_checkpoint, taco_hparams, args.waveglow_checkpoint)