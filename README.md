# AI-scriptie
By Lucas Bron, 4757092

The Jup file is the main part of the code. The main.py is an old file that was used to test and to put the GWC data into the right directories.

Log of steps I took to get this code to work:

Cuda python wheels
-cuda prep: pip install --upgrade setuptools pip wheel
-cuda12 runtime: pip install nvidia-cuda-runtime-cu12
whisperx 3.8.6
-pip install whisper
-pip install faster-whisper
pip install git+https://github.com/m-bain/whisperx.git

fix error (https://github.com/SYSTRAN/faster-whisper/issues/1589)
-pip install --force-reinstall --no-deps git+https://github.com/SYSTRAN/faster-whisper.git

-pip install hf_xet

download from https://www.fdr.uni-hamburg.de/record/1875 
extract swc-en en swc-german into root_dir/input

copy of de-en/wav.txt into input map to edit

pip install pydub

Runtimewarning ffmpeg not found with error File not found, solution:
ffmpeg download from https://www.gyan.dev/ffmpeg/builds/#tools git master build essential
add /bin of ffmpeg to local user environment variables

pip install evaluate
pip install jiwer

ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module, but found huggingface-hub==2.2.0.
Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main
pip install transformers -U

ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` requires `accelerate`. You can install it with `pip install accelerate`
pip install accelerate

pip install speechbrain
Put on developer mode to avoid winerror 1314
en local_strategy=LocalStrategy.COPY for windows compatability

pip install nltk

pip install jupyter

pip install num2words

pip install -U cuda-pathfinder

(https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec)
pip uninstall -y torch torchvision torchaudio torchcodec
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install torch==0.7.0 

ffmpeg-8.1.2-fullbuildshard installed and added to PATH user envirment variable. then again other version to match.
pip uninstall -y torch torchvision torchaudio torchcodec
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install torchcodec==0.9.0 --index-url https://download.pytorch.org/whl/cu126
