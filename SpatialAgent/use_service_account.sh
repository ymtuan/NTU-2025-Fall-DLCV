#!/bin/bash
# 使用服務帳戶金鑰文件設置認證
# 使用方法: source use_service_account.sh <path_to_key.json>

if [ -z "$1" ]; then
    echo "錯誤: 請提供服務帳戶金鑰文件路徑"
    echo ""
    echo "使用方法:"
    echo "  source use_service_account.sh /path/to/your/service-account-key.json"
    echo ""
    echo "或者直接設置環境變數:"
    echo "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json"
    return 1
fi

KEY_FILE="$1"

if [ ! -f "$KEY_FILE" ]; then
    echo "錯誤: 文件不存在: $KEY_FILE"
    return 1
fi

# 設置環境變數
export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE"
echo "✓ 已設置 GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE"
echo ""
echo "現在可以執行 agent_run.py 了："
echo "  cd agent"
echo "  python agent_run.py --project_id gen-lang-client-0164359804"
