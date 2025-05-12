{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems = {
      url = "github:nix-systems/default";
    };
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };

  outputs = {nixpkgs, ...} @ inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      perSystem = {system, ...}: let
        pkgs = import nixpkgs {inherit system;};
      in {
        devShells.default = pkgs.mkShell {
          venvDir = ".venv";
          postShellHook = ''pip install -r requirements.txt'';
          strictDeps = false;
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.gurobi}/lib:$LD_LIBRARY_PATH
            export GUROBI_HOME=${pkgs.gurobi}
            export GUROBI_VERSION=$(basename $(ls -d ${pkgs.gurobi}) | sed 's/.*-\([0-9]\+\)\.\([0-9]\+\).*/\1\2/')
            export GRB_LICENSE_FILE=/home/prometheus/.config/gurobi/gurobi.lic
          '';
          packages = with pkgs.python312Packages;
            [
              matplotlib
              moviepy
              flake8
              rope
              isort
              ipython
              scipy
              seaborn
              ruff
              pyside6
              jupyterlab
              gymnasium
              networkx
              pybox2d
              pygame
              pkgs.gurobi
              # torch-bin
              # opencv4
              # pkgs.cudaPackages.cudatoolkit
              # pkgs.texliveFull
            ]
            ++ (with pkgs; []);
        };
      };
    };
}
