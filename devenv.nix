{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  talib = pkgs.stdenv.mkDerivation rec {
    pname = "ta-lib";
    version = "0.6.2";
    src = pkgs.fetchFromGitHub {
      owner = "TA-Lib";
      repo = "ta-lib";
      rev = version;
      sha256 = "sha256-bIzN8f9ZiOLaVzGAXcZUHUh/v9z1U+zY+MnyjJr1lSw=";
    };

    nativeBuildInputs = with pkgs; [
      pkg-config
      autoreconfHook
    ];
    hardeningDisable = ["format"];

    meta = with lib; {
      description = "TA-Lib is a library that provides common functions for the technical analysis of financial market data.";
      mainProgram = "ta-lib-config";
      homepage = "https://ta-lib.org/";
      license = lib.licenses.bsd3;

      platforms = platforms.linux;
      maintainers = with maintainers; [rafael];
    };
  };
in {
  name = "ft_userdata";

  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages = with pkgs; [
    git
    poetry
    talib
  ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;
  languages = {
    nix = {
      enable = true;
    };
    shell = {
      enable = true;
    };
    python = {
      enable = true;
      # version = "3.12";
      package = pkgs.python312;
      libraries = [
        "${config.devenv.dotfile}/profile"
        # pkgs.python312Packages.ta-lib
        talib
      ];
      poetry = {
        enable = true;
        activate = {
          enable = true;
        };
        install = {
          enable = true;
        };
      };
    };
  };

  cachix = {
    enable = true;
  };

  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  enterShell = ''
    hello
    git --version
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/
    export TA_LIBRARY_PATH=${talib}/lib
    export TA_INCLUDE_PATH=${talib}/include
  '';

  # https://devenv.sh/tasks/
  tasks = {
    # "bin:install-local" = {
    #   exec = "bin/install-local";
    # };
    # "devenv:enterShell" = {
    #   after = ["bin:install-local"];
    # };
    "bash:shellHook" = {
      exec = "LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/";
      before = ["devenv:enterShell" "devenv:enterTest"];
    };
  };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
