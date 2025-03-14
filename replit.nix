
{pkgs}: {
  deps = [
    pkgs.libxcrypt
    pkgs.eigen
    pkgs.catch
    pkgs.gcc
    pkgs.cmake
    pkgs.python311Packages.pybind11
  ];
}
