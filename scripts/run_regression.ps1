param(
  [int]$Windows = 8
)

$ErrorActionPreference = 'Stop'

python .\tools\export_tflite_artifacts.py --tflite data\model_int8.tflite --outdir data
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python .\tools\rtl_vs_tflite_regression.py --repo . --tflite data\model_int8.tflite --input input_data.mem --windows $Windows --raw-mgdl
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
