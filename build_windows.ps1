# Build script for ORB-SLAM3 on Windows with MSVC + vcpkg
# Usage: .\build_windows.ps1 [-VcpkgRoot <path>] [-BuildType Release|Debug] [-Jobs <n>]
param(
    [string]$VcpkgRoot = "C:\vcpkg",
    [string]$BuildType = "Release",
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"

$VS_BASE  = "C:\Program Files\Microsoft Visual Studio\2022\Professional"
$VS_CMAKE = "$VS_BASE\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$VS_NINJA = "$VS_BASE\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

if (-not (Test-Path $VS_CMAKE)) {
    Write-Error "cmake not found at: $VS_CMAKE`nInstall Visual Studio 2022 with 'C++ CMake tools for Windows' component."
    exit 1
}

# --- Add VS-bundled Ninja and compiler to PATH ---
# The VS generator finds cl.exe automatically, but we expose Ninja for reference.
$env:PATH = "$VS_BASE\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;" + `
            "$VS_BASE\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64;" + `
            $env:PATH

# --- vcpkg ---
if (-not (Test-Path "$VcpkgRoot\vcpkg.exe")) {
    Write-Host "vcpkg not found at $VcpkgRoot. Cloning and bootstrapping..."
    git clone https://github.com/microsoft/vcpkg.git $VcpkgRoot
    & "$VcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
}

$toolchain = "$VcpkgRoot\scripts\buildsystems\vcpkg.cmake"

# --- Extract ORB vocabulary ---
$vocabGz  = Join-Path $PSScriptRoot "Vocabulary\ORBvoc.txt.tar.gz"
$vocabTxt = Join-Path $PSScriptRoot "Vocabulary\ORBvoc.txt"
if (-not (Test-Path $vocabTxt)) {
    Write-Host "Extracting ORB vocabulary..."
    if (Get-Command tar -ErrorAction SilentlyContinue) {
        tar -xf $vocabGz -C (Join-Path $PSScriptRoot "Vocabulary")
    } else {
        Write-Warning "tar not found. Please extract Vocabulary\ORBvoc.txt.tar.gz manually."
    }
}

# --- Configure ---
# Use the "Visual Studio 17 2022" generator: cmake finds cl.exe automatically
# without needing vcvars64.bat. Multi-config; build type is set at build time.
$buildDir = Join-Path $PSScriptRoot "build_msvc"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$cmakeArgs = @(
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
    "-DVCPKG_TARGET_TRIPLET=x64-windows",
    ".."
)

Write-Host "`n=== Configuring ===`n"
Push-Location $buildDir
& $VS_CMAKE @cmakeArgs
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

# --- Build ---
$buildArgs = @("--build", ".", "--config", $BuildType, "--parallel")
if ($Jobs -gt 0) { $buildArgs += $Jobs }

Write-Host "`n=== Building ($BuildType) ===`n"
& $VS_CMAKE @buildArgs
$rc = $LASTEXITCODE
Pop-Location

if ($rc -eq 0) {
    Write-Host "`nBuild succeeded. Binaries are in Examples/*/$BuildType/ subdirectories."
} else {
    Write-Error "Build failed with exit code $rc"
    exit $rc
}
