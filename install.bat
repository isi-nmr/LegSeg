@echo off
REM Installation script for the LegSeg GUI (Windows).
REM Usage: install.bat         (CUDA 12.8)
REM        install.bat cpu     (CPU-only torch)
setlocal
cd /d "%~dp0"

set CUDA=%1
if "%CUDA%"=="" set CUDA=cu128

REM 1) venv with Python 3.14
py -3.14 -m venv .venv || exit /b 1
set PY=.venv\Scripts\python.exe
%PY% -m pip install --upgrade pip || exit /b 1

REM 2) torch (must be installed before nnU-Net)
%PY% -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CUDA% || exit /b 1

REM 3) nnU-Net and GUI dependencies
%PY% -m pip install -r requirements.txt || exit /b 1

REM 4) trained model (from GitHub Release)
set MODEL_URL=https://github.com/isi-nmr/LegSeg/releases/download/model-v1/Dataset015_SARXfinal.zip
if not exist nnUNet_results\Dataset015_SARXfinal (
    if not exist nnUNet_results mkdir nnUNet_results
    curl -L --fail -o model.zip %MODEL_URL% || exit /b 1
    tar -xf model.zip -C nnUNet_results || exit /b 1
    del model.zip
)

echo Done. Run the GUI with: .venv\Scripts\python.exe legseg.py
endlocal
