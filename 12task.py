import time
from collections import deque
import threading

def capture_strings(input_queue, stop_event):
    """持续接收用户输入的字符串并存储"""
    print("请开始输入字符串（按Ctrl+C停止输入）：")
    try:
        while not stop_event.is_set():
            user_input = input()
            if user_input:
                # 存储输入内容和对应的时间戳
                input_queue.append((time.time(), user_input))
    except KeyboardInterrupt:
        print("\n输入结束")

def get_last_30s_strings(input_queue):
    """获取最后30秒内输入的字符串"""
    current_time = time.time()
    thirty_seconds_ago = current_time - 30
    
    # 筛选出30秒内的输入
    recent_entries = [entry for entry in input_queue if entry[0] >= thirty_seconds_ago]
    
    # 提取字符串内容并拼接
    return ' '.join([entry[1] for entry in recent_entries])

def main():
    # 使用双端队列存储输入（带时间戳）
    input_queue = deque()
    stop_event = threading.Event()
    
    # 启动输入捕获线程
    capture_thread = threading.Thread(
        target=capture_strings,
        args=(input_queue, stop_event),
        daemon=True
    )
    capture_thread.start()
    
    try:
        # 等待用户触发key（这里用输入"key"作为触发信号）
        while True:
            trigger = input("\n输入'key'以截取最后30秒的内容: ")
            if trigger.lower() == 'key':
                result = get_last_30s_strings(input_queue)
                print("\n最后30秒的输入内容：")
                print("------------------------")
                print(result)
                print("------------------------")
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        capture_thread.join()

if __name__ == "__main__":
    main()
