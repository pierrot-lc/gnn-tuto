{
  description = "Jax devshell";

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
      "https://ploop.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "ploop.cachix.org-1:i6+Fqarsbf5swqH09RXOEDvxy7Wm7vbiIXu4A9HCg1g="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = false;
      };
    };

    python-packages = ps:
      with ps; [
        pip
        setuptools
        virtualenv
      ];

    fhs = pkgs.buildFHSUserEnv {
      name = "jax";
      targetPkgs = pkgs: [
        (pkgs.python312.withPackages python-packages)
        pkgs.just
        pkgs.uv
      ];
      multiPkgs = pkgs: [
        pkgs.zlib # Numpy dep.
      ];
    };
  in {
    devShells.${system}.default = fhs.env;
  };
}
