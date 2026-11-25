{
  description = "Ling 3.2 Air text model for Gufo";

  # Gufo v0.36.0, the release this extension targets (extension API 0.3.x).
  inputs.gufo.url = "github:gufo-org/gufo/4a7d8c6d178fc6979e5ae38c07b40e1fe89dd1ce";

  outputs = { self, gufo }: {
    # A Gufo with this extension compiled in. Development builds use the pinned
    # Gufo checkout's own SDK instead (`gufo extension install`), so both routes
    # stay available.
    packages.x86_64-linux.default =
      gufo.packages.x86_64-linux.default.override { extensions = [ self ]; };

    # Gufo plus its extension SDK, for building this repository as a module.
    packages.x86_64-linux.sdk = gufo.packages.x86_64-linux.sdk;
  };
}
