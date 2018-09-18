#!/usr/bin/env bash
if [[ -z $1 ]]; then
  echo "Must provide an image URL."
  exit 1
fi

extract-keypoints -v --image-url $1 | detect-features
