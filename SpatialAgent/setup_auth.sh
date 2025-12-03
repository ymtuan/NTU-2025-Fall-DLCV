#!/bin/bash
# 設置 Google Cloud 認證
# 使用方法：source setup_auth.sh <path_to_service_account_key.json>

if [ -z "$1" ]; then
    echo "使用方法: source setup_auth.sh <path_to_service_account_key.json>"
    echo "或者設置環境變數:"
    echo "export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json"
else
    export GOOGLE_APPLICATION_CREDENTIALS="$1"
    echo "已設置 GOOGLE_APPLICATION_CREDENTIALS=$1"
fi
