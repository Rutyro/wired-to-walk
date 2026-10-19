param(
    [string]$Dest = (Join-Path $PSScriptRoot "..\data\malecns")
)

$ErrorActionPreference = "Stop"
$base = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
$files = @(
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

foreach ($name in $files) {
    $path = Join-Path $Dest $name
    if (Test-Path $path) {
        Write-Host "exists: $name"
        continue
    }

    Write-Host "download: $name"
    & curl.exe -L --fail --retry 3 --continue-at - "$base/$name" -o $path
    if ($LASTEXITCODE -ne 0) {
        throw "download failed: $name"
    }
}

Write-Host "MaleCNS tables are in $Dest"
Write-Host "License: CC BY 4.0 - https://creativecommons.org/licenses/by/4.0/"
