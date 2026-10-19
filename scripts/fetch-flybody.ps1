param(
    [string]$Dest = (Join-Path $PSScriptRoot "..\data\flybody")
)

$ErrorActionPreference = "Stop"
$repo = "https://github.com/TuragaLab/flybody.git"
$tmp = Join-Path $env:TEMP ("flybody-" + [guid]::NewGuid().ToString("N"))

try {
    git clone --depth 1 --filter=blob:none --sparse $repo $tmp
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed"
    }

    git -C $tmp sparse-checkout set flybody/fruitfly/assets
    if ($LASTEXITCODE -ne 0) {
        throw "git sparse-checkout failed"
    }

    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    $assets = Join-Path $tmp "flybody/fruitfly/assets"
    Get-ChildItem -Path $assets -File |
        Where-Object { $_.Extension -in ".xml", ".obj" } |
        Copy-Item -Destination $Dest -Force

    Copy-Item (Join-Path $tmp "LICENSE") (Join-Path $Dest "LICENSE") -Force
    Write-Host "flybody assets are in $Dest (Apache-2.0)"
}
finally {
    if (Test-Path $tmp) {
        Remove-Item -Recurse -Force $tmp
    }
}
