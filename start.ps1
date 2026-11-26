param(
    [switch]$Evaluate
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonCommand = Get-Command python -ErrorAction Stop
    & $pythonCommand.Source -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the local Python environment."
    }
}

$probe = @'
import os
import cv2
import face_recognition
import face_recognition_models as models
import flask
import joblib
import numpy
import PIL
import scipy
assert os.path.isfile(models.pose_predictor_model_location())
'@
$probe | & $venvPython -
$dependenciesReady = $LASTEXITCODE -eq 0

if (-not $dependenciesReady) {
    & $venvPython -m pip install --upgrade pip "setuptools<81"
    if ($LASTEXITCODE -ne 0) { throw "Could not prepare pip and setuptools." }

    & $venvPython -m pip install --no-deps -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install project requirements." }

    & $venvPython -m pip install flask flask-cors werkzeug opencv-python numpy Pillow joblib scipy dlib-bin
    if ($LASTEXITCODE -ne 0) { throw "Could not install the application dependencies." }

    $modelsArchive = Join-Path $env:TEMP "face_recognition_models_upstream.zip"
    Invoke-WebRequest `
        -Uri "https://github.com/ageitgey/face_recognition_models/archive/refs/heads/master.zip" `
        -OutFile $modelsArchive
    if ($LASTEXITCODE -ne 0) { throw "Could not download the upstream face-recognition models." }

    & $venvPython -m pip install $modelsArchive
    if ($LASTEXITCODE -ne 0) { throw "Could not install the upstream face-recognition models." }
    Remove-Item -LiteralPath $modelsArchive -Force

    $probe | & $venvPython -
    if ($LASTEXITCODE -ne 0) { throw "The Python environment is missing a required application dependency." }
}

if ($Evaluate) {
    $runAllScript = Join-Path $PSScriptRoot "run_all.py"
    & $venvPython $runAllScript --evaluate --skip-req-check
    if ($LASTEXITCODE -ne 0) {
        throw "Model evaluation failed; see evaluation_split.log for details."
    }
}

$env:FLASK_HOST = "127.0.0.1"
$env:FLASK_DEBUG = "0"
& $venvPython app.py
exit $LASTEXITCODE
