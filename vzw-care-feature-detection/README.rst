==========================
vzw-care-feature-detection
==========================

Feature detection for vzw-care-tf-pose data


Description
===========

Install for dev with

  pip install -e .

in the library working directory.

This creates one entrypoint 'detect-features'. To run with data, send a
json list (list of json objects, not a json list) in to the entry. i.e.:

  detect-features < data.jsonl > outputs.jsonl

Use the -v or -vv flags for logging to stderr.

Note
====
This project has been set up using PyScaffold 3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
