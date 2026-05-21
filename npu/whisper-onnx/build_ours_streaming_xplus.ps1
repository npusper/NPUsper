param(
    [Parameter(Mandatory = $true)]
    [string]$OrtRoot,

    [string]$BuildDir = "build_xplus",

    [string]$Config = "Release",

    [string]$DeployDir = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildPath = Join-Path $root $BuildDir

Write-Host "Source: $root"
Write-Host "Build:  $buildPath"
Write-Host "ORT:    $OrtRoot"

$clFlags = $env:CL
if (-not $clFlags) {
    $clFlags = ""
}
$clFlags = $clFlags.Trim()
if ($clFlags -notmatch "(^|\s)/utf-8(\s|$)") {
    $env:CL = ("/utf-8 " + $clFlags).Trim()
}

if (-not (Test-Path -LiteralPath $OrtRoot)) {
    throw "OrtRoot does not exist: $OrtRoot"
}

$headerCandidates = @(
    (Join-Path $OrtRoot "build\native\include\onnxruntime_cxx_api.h"),
    (Join-Path $OrtRoot "include\onnxruntime_cxx_api.h"),
    (Join-Path $OrtRoot "include\onnxruntime\core\session\onnxruntime_cxx_api.h")
)
$libCandidates = @(
    (Join-Path $OrtRoot "runtimes\win-arm64\native\onnxruntime.lib"),
    (Join-Path $OrtRoot "lib\onnxruntime.lib")
)
$nativeDir = Join-Path $OrtRoot "runtimes\win-arm64\native"

if (-not ($headerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1)) {
    throw "ONNX Runtime C++ headers were not found under OrtRoot. Use the extracted Microsoft.ML.OnnxRuntime.QNN NuGet package, not the Python pip package."
}
if (-not ($libCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1)) {
    throw "onnxruntime.lib was not found under OrtRoot. Use the win-arm64 Microsoft.ML.OnnxRuntime.QNN NuGet package."
}

cmake -S $root -B $buildPath -DORT_ROOT="$OrtRoot"
if ($LASTEXITCODE -ne 0) {
    throw "cmake configure failed"
}

cmake --build $buildPath --config $Config --target ours_streaming
if ($LASTEXITCODE -ne 0) {
    throw "cmake build failed"
}

$candidates = @(
    (Join-Path $buildPath "bin\ours_streaming.exe"),
    (Join-Path $buildPath "bin\$Config\ours_streaming.exe"),
    (Join-Path $buildPath "$Config\ours_streaming.exe")
)

$exe = $null
foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
        $exe = $candidate
        break
    }
}

if (-not $exe) {
    throw "ours_streaming.exe not found under $buildPath"
}

Write-Host "Built:  $exe"

if ($DeployDir) {
    if (-not (Test-Path -LiteralPath $DeployDir)) {
        New-Item -ItemType Directory -Force $DeployDir | Out-Null
    }
    $deployPath = (Resolve-Path -LiteralPath $DeployDir).Path
    $dst = Join-Path $deployPath "ours_streaming.exe"
    Copy-Item -LiteralPath $exe -Destination $dst -Force
    Write-Host "Copied: $dst"

    if (Test-Path -LiteralPath $nativeDir) {
        Copy-Item -Path (Join-Path $nativeDir "*.dll") -Destination $deployPath -Force
        Write-Host "Copied ONNX Runtime/QNN DLLs from: $nativeDir"
    }
}
