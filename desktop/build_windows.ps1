param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$desktopRoot = $PSScriptRoot
$taskPython = (Get-Command $Python -CommandType Application -ErrorAction Stop).Source
$taskOriginalPath = $env:PATH
try {
    # Avoid accidentally bundling incompatible ICU/OpenSSL DLLs from other tools.
    $env:PATH = "$(Split-Path -Parent $taskPython);$env:SystemRoot\System32;$env:SystemRoot"
    & $taskPython -m PyInstaller --noconfirm --clean --windowed --onedir --name SignalScope --distpath (Join-Path $desktopRoot 'dist') --workpath (Join-Path $desktopRoot 'build') --specpath (Join-Path $desktopRoot 'build') --paths $desktopRoot --paths (Split-Path -Parent $desktopRoot) --add-data "$(Join-Path $desktopRoot '../models/modulation/scanner_*.npz');models/modulation" --add-data "$(Join-Path $desktopRoot '../models/modulation/scanner_gb*.joblib');models/modulation" --hidden-import sklearn.ensemble._gb --hidden-import sklearn.tree._classes --hidden-import sklearn.dummy --collect-submodules pyvisa_py --copy-metadata PyVISA-py (Join-Path $desktopRoot 'run.py')
    if ($LASTEXITCODE -ne 0) { throw 'Windows build failed' }
} finally {
    $env:PATH = $taskOriginalPath
}
Write-Output "Portable app: $(Join-Path $desktopRoot 'dist\SignalScope\SignalScope.exe')"
