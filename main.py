import whisperx
from faster_whisper import WhisperModel
import logging
import tarfile
import zipfile
from pathlib import Path

#open archive to acces audiofiles
root_dir = Path(__file__).parent
tar_path = root_dir / "input\\SWC_German.tar"

print("reading and creating filter...")
a = []
sf = root_dir / "input\\selected_files.txt"
with open(sf, "r") as sf:
    for line in sf:
        f = f"german/{line.split(' ')[1]}"
        a.append(f.strip())

print("reading and filtering...")
with tarfile.open(tar_path, mode='r|*') as geTar:
    geMemb = geTar.getmembers()
    geMemb2 = [tarItem for tarItem in geMemb if tarItem.name in a]

print("extracting...")
extract_path = root_dir / "input\\code-switched-audio\\"
with tarfile.open(tar_path, mode='r|*') as geTar2:
    geTar2.extractall(path=extract_path, members=geMemb2)

print("done")

#code to create selected_files-file for code-switched corpus collection

#code-switched corpus collection file writer
geDict = {"number":"name"}
selec_file = root_dir / "input\\wav_copy.scp"
with open(selec_file, "r") as sf:
    for line in sf:
        ls = line.split(' ')
        val = ls[2][15:]
        geDict[ls[0]] = val

newFile = root_dir / "input\\selected_files.txt"
with open(newFile, "w") as nf:
    for k,v in geDict.items():
        nf.write(f"{k}: {v}\n")

#toycode to test faster whipser on test.mp3 file
print("Start...")

device = "cpu"
batch_size = 8
compute_type = "int8"
model_size = "tiny"
audio_file = "test2minute.mp3"
model_dir = root_dir / "models"
model_path = model_dir / "models--Systran--faster-whisper-tiny"
output_dir = "outputs"
output_file = output_dir + "\\test_output.txt"

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.INFO)

print("loading model...")
model = WhisperModel(model_size, device, compute_type=compute_type, download_root=model_dir)

print("model loaded, loading audio & transcribing")
segments, info = model.transcribe(audio_file, beam_size=5)

print("writing to output file:")
with open(output_file, "a") as f:
    f.write(f"Model_name: Faster-Whisper_toy. size = {model_size}, batch size = {batch_size}, compute type = {compute_type}, audio_file = {audio_file} \n")
    
    f.write("Start. \n")
    for segment in segments:
        f.write("[%.2fs -> %.2fs] %s \n" % (segment.start, segment.end, segment.text))
    f.write("End.")

print(logging.log)

print("Done!")