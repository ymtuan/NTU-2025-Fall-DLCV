#!/usr/bin/env python3
"""
測試 vLLM 連接
"""

import sys
from llm_client import create_llm_client

def test_vllm_connection(api_base="http://localhost:8040/v1", model_name="Qwen/Qwen3-4B-Instruct-2507"):
    """測試 vLLM 連接"""
    print(f"測試連接到: {api_base}")
    print(f"模型: {model_name}")
    
    try:
        client = create_llm_client(
            client_type='openai',
            api_base=api_base,
            api_key='dummy',
            model=model_name,
            temperature=0.2,
            max_tokens=100
        )
        
        print("\n發送測試訊息...")
        response = client.send_message("Hello! Please respond with 'OK' if you can hear me.")
        print(f"回應: {response}")
        
        print("\n✓ vLLM 連接成功！")
        return True
        
    except Exception as e:
        print(f"\n✗ 連接失敗: {e}")
        print("\n請確認:")
        print("1. vLLM 服務器是否正在運行")
        print("2. API base URL 是否正確")
        print("3. 模型名稱是否正確")
        print("\n啟動 vLLM 服務器的命令:")
        print("CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \\")
        print("    --model Qwen/Qwen3-4B-Instruct-2507 \\")
        print("    --max-model-len 4096 \\")
        print("    --host 0.0.0.0 \\")
        print("    --gpu-memory-utilization 0.7 \\")
        print("    --port 8040")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_base', type=str, default='http://localhost:8040/v1')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B-Instruct-2507')
    args = parser.parse_args()
    
    success = test_vllm_connection(args.api_base, args.model)
    sys.exit(0 if success else 1)

