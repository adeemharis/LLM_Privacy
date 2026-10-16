"""Backward-compatible entry point for publishing all Assignment 1 adapters."""
try:
    from .hf_adapters import main
except ImportError:
    from hf_adapters import main

if __name__ == "__main__":
    import sys
    # This shim defaults to the publisher subcommand, but accepts it explicitly too.
    if len(sys.argv) > 1 and sys.argv[1] == "publish":
        sys.argv.pop(1)
    sys.argv.insert(1, "publish")
    main()
