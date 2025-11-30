# CudaConvoLab

Repository: https://github.com/Shareef91/CudaConvoLab

A small CUDA assignment demonstrating a parallel masked moving average (convolution-like) implemented in `pAverage.cu`. This repository contains a sequential reference (`sAverage.cu`) and a parallel CUDA implementation that uses constant memory for the mask and a strided-thread pattern to compute the output.

## Files

- `pAverage.cu` — CUDA implementation using `__constant__` memory for the mask and a strided kernel.
- `sAverage.cu` — sequential C implementation (reference implementation).
- `input.txt` — example input file (first line is `n`, followed by `n` floats).
- `output.txt` — example output produced by running the program.

## Description

The program applies a fixed-size mask to an input 1D signal to produce an averaged output. The mask is stored in constant memory on the device for fast reads. Edge handling mirrors the sequential version: negative indices are clamped to `input[0]`.

## Build (Windows, PowerShell)

You need the NVIDIA CUDA Toolkit and a host MSVC compiler installed (Visual Studio or Build Tools).

Option A — Developer Command Prompt for Visual Studio (recommended):

```powershell
cd '<WORKSPACE_PATH>'
nvcc -o pAverage.exe pAverage.cu
```

Option B — PowerShell with Visual Studio environment initialized (adjust path for your VS version):

```powershell
& 'C:\Path\To\VisualStudio\vcvars64.bat';
cd '<WORKSPACE_PATH>';
nvcc -o pAverage.exe pAverage.cu
```

If compilation succeeds, an executable `pAverage.exe` will be produced.

## Run

Run the program with an input file and output file path:

```powershell
.\pAverage.exe input.txt output.txt
```

`input.txt` format:

```
<n>
<float 1>
<float 2>
...
<float n>
```

`output.txt` will contain `n` on the first line and the `n` averaged values (one per line) with 3 decimal places.

## Notes & Tips

- If you get `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`, open the Developer Command Prompt for Visual Studio or run the `vcvars` script to set up `cl.exe` in your environment.
 - If you get `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`, open the Developer Command Prompt for Visual Studio or run the `vcvars` script (example: `C:\Path\To\VisualStudio\vcvars64.bat`) to set up `cl.exe` in your environment.
- You can tune kernel launch parameters in `pAverage.cu` (change `threads` and `blocks` values) to experiment with occupancy and performance.
- Consider adding CUDA error-check macros around `cudaMalloc`, `cudaMemcpy`, and kernel launches to get clearer runtime diagnostics.

## Notes on Repository Naming

The repository currently uses the name shown at the top. If you prefer a different name, rename the repo on GitHub or create a new repo and push the code there.

## License

MIT License — feel free to adapt for your coursework. Include your own name and institution if required.

---
Contributions welcome — feel free to open issues or submit pull requests. Enjoy experimenting with parallel convolution-like kernels!
