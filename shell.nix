{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    # 1. Base Python Tools
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.virtualenv
    
    # 2. Libraries required by your script
    pkgs.python3Packages.netcdf4
    pkgs.python3Packages.numpy
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.scipy

    # 3. Interactive/Jupyter support
    # VSCodium explicitly needs 'notebook' and 'jupyter' to run cells, not just ipykernel
    pkgs.python3Packages.ipykernel
    pkgs.python3Packages.jupyter
    pkgs.python3Packages.notebook

    # 4. Editor
    pkgs.vscodium
  ];

  shellHook = ''
    # 1. Create virtualenv if missing
    if [ ! -d ".venv" ]; then
      echo "Creating new virtual environment..."
      # CRITICAL: --system-site-packages allows the venv to use the 
      # heavy libraries (scipy, numpy, notebook) installed via Nix.
      python -m venv .venv --system-site-packages
    fi

    # 2. Activate
    source .venv/bin/activate

    # 3. Register the kernel for VSCodium
    python -m ipykernel install --user --name=plot_env --display-name "Python (Plotting Env)"

    echo "Environment ready with Scipy/Matplotlib. Launching VSCodium..."
    codium . & 
  '';
}
