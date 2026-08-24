#!/usr/bin/env bash
set -eu
kubectl rollout status deployment/cart
