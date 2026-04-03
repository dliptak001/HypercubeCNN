# How to Build (CLI)

cmake and g++ are bundled with CLion and are NOT on the system PATH.

## Toolchain Paths

| Tool | Path |
|------|------|
| CMake | `C:\Program Files\JetBrains\CLion 2024.3.2\bin\cmake\win\x64\bin\cmake.exe` |
| Ninja | `C:\Program Files\JetBrains\CLion 2024.3.2\bin\ninja\win\x64\ninja.exe` |
| MinGW | `C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin\` |

## Build Commands

Bash cannot capture g++ stderr (it goes to a Windows console handle). You must use PowerShell.

### Release Build

```bash
powershell.exe -File - <<'PS1'
$cmake = 'C:\Program Files\JetBrains\CLion 2024.3.2\bin\cmake\win\x64\bin\cmake.exe'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
& $cmake --build C:\CLion\HypercubeCNNStaging\cmake-build-release 2>&1
PS1
```

### Debug Build

```bash
powershell.exe -File - <<'PS1'
$cmake = 'C:\Program Files\JetBrains\CLion 2024.3.2\bin\cmake\win\x64\bin\cmake.exe'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
& $cmake --build C:\CLion\HypercubeCNNStaging\cmake-build-debug 2>&1
PS1
```

## Running the Executable

Diagnostic executables depend on libgomp-1.dll (OpenMP runtime). Set MinGW on PATH before running:

```bash
powershell.exe -File - <<'PS1'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
& "C:\CLion\HypercubeCNNStaging\cmake-build-release\HypercubeCNN.exe" 2>&1
PS1
```

## Important Notes

- **Do not reconfigure cmake-build-\* directories.** CLion owns them. Running `cmake -B` with `-G` flags will break the IDE integration.
- If the build directory is missing or broken, delete it and reload CMake from CLion (File > Reload CMake Project).
- Prefer Release mode for tests and diagnostics (Debug uses different float behavior with -ffast-math).
