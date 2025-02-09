{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  talib = pkgs.stdenv.mkDerivation rec {
    pname = "ta-lib";
    version = "0.6.1";
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
    python = {
      enable = true;
      libraries = [
        "${config.devenv.dotfile}/profile"
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
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
