# TensorFlow Crail

```sh
$ curl -OL https://github.com/bazelbuild/bazel/releases/download/0.20.0/bazel-0.20.0-installer-linux-x86_64.sh
$ chmod +x bazel-0.20.0-installer-linux-x86_64.sh
$ ./bazel-0.20.0-installer-linux-x86_64.sh
$ ./configure.sh
$ bazel build build_pip_pkg
$ bazel-bin/build_pip_pkg artifacts
```

A package file `artifacts/tensorflow_crail-*.whl` will be generated after a build is successful.


