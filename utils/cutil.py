import csv
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from typing import Tuple, Optional, Union
import json

import httpx

# 读取文件内容
def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.info(f"读取文件失败: {e}")
        return ""

# 检查本地存储文件夹是否有权限
def store_dir_access() -> bool:
    try:
        home = Path.home()
        if not os.access(home, os.W_OK):
            logging.error(f"path: {home} 无权限写入")
            return False
        dlt_dir = home / ".dlt_analysis"
        if not dlt_dir.exists():
            try:
                dlt_dir.mkdir(parents=True)
            except Exception:
                logging.error(f"path: {dlt_dir} 创建文件夹失败")
                return False
        if not os.access(dlt_dir, os.W_OK):
            logging.error(f"path: {dlt_dir} 无权限写入")
            return False
        return True
    except Exception:
        return False


# 列出本地存储文件夹的列表
def store_dir_list() -> List[str]:
    dlt_dir = Path.home() / ".dlt_analysis"
    if not dlt_dir.exists() or not dlt_dir.is_dir():
        return []

    files = [
        (f.name[:-8], f.stat().st_ctime)
        for f in dlt_dir.iterdir()
        if f.is_file() and f.name.endswith(".dlt.csv")
    ]

    files.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in files]

def check_file_exist(file_name) -> bool:
    if not store_dir_access():
        return False
    filepath = Path.home() / ".dlt_analysis" / (file_name + ".dlt.csv")
    if filepath.exists():
        return True
    else:
        return False

def get_abspath(file_name):
    return Path.home() / ".dlt_analysis" / (file_name + ".dlt.csv")

def save_to_store_file(data: list[dict]) -> bool:
    try:
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        first_issue = data[-1].get("issue", "unknown")
        last_issue = data[0].get("issue", "unknown")
        file_name = f"{now}_{first_issue}_{last_issue}"
        filepath = Path.home() / ".dlt_analysis" / (file_name + ".dlt.csv")

        headers = ["issue", "rb_1", "rb_2", "rb_3", "rb_4", "rb_5", "bb_1", "bb_2", "date"]

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in data:
                writer.writerow({key: row.get(key, "") for key in headers})
        return file_name
    except Exception as e:
        logging.error(f"存储文件失败{e}")
        return ""

def delete_store_file(file_name) -> bool:
    try:
        file_path = Path.home() / ".dlt_analysis" / (file_name + ".dlt.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except Exception as e:
        logging.error(f"删除文件失败{e}")
        return False


# 范围搜索
def search(start,end) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Referer": "https://datachart.500.com/dlt/history/history.shtml",
        "X-Requested-With": "XMLHttpRequest",
    }

    url = f"https://datachart.500.com/dlt/history/newinc/history.php?start={start}&end={end}"
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(url,headers=headers)
            if response.status_code == 200:
                logging.info(f"{url} 搜索成功响应: {response.text}")
                return response.text
            else:
                logging.error(f"{url} 搜索失败响应: {response}")
                return ""
    except Exception as e:
        logging.error(f"{url} 搜索异常响应: {response}")
        return ""
# 最近搜索
def search_limit(limit) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Referer": "https://datachart.500.com/dlt/history/history.shtml",
        "X-Requested-With": "XMLHttpRequest",
    }
    url = f"https://datachart.500.com/dlt/history/newinc/history.php?limit={limit}&sort=0"
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(url,headers=headers)
            if response.status_code == 200:
                logging.info(f"{url} 搜索成功响应: {response.text}")
                return response.text
            else:
                logging.error(f"{url} 搜索失败响应: {response}")
                return ""
    except Exception as e:
        logging.error(f"{url} 搜索异常响应: {response}")
        return ""

# 爬虫数据解析
def response_html_to_dict(content: str) -> list[dict]:
    soup = BeautifulSoup(content, "html.parser")
    rows = soup.select("#tdata tr")
    result = []

    try:
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 15:
                data = {
                    "issue": cols[0].text.strip(),
                    "rb_1": cols[1].text.strip(),
                    "rb_2": cols[2].text.strip(),
                    "rb_3": cols[3].text.strip(),
                    "rb_4": cols[4].text.strip(),
                    "rb_5": cols[5].text.strip(),
                    "bb_1": cols[6].text.strip(),
                    "bb_2": cols[7].text.strip(),
                    "date": cols[14].text.strip(),
                }
                result.append(data)
        return result
    except Exception as e:
        logging.error(f"解析html响应数据报错 {e}")
        return []


# 加载本地存储的文件到表格字典
def store_file_to_dict(file_name: str) -> list[dict]:
    try:
        path = Path.home() / ".dlt_analysis" / (file_name + ".dlt.csv")
        file_path = Path(path)
        if not file_path.exists():
            logging.error(f"解析本地csv数据至表格失败,csv文件 {path} 不存在")

        result = []
        with open(file_path, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data = {
                    "issue": row.get("issue", ""),
                    "rb_1": row.get("rb_1", ""),
                    "rb_2": row.get("rb_2", ""),
                    "rb_3": row.get("rb_3", ""),
                    "rb_4": row.get("rb_4", ""),
                    "rb_5": row.get("rb_5", ""),
                    "bb_1": row.get("bb_1", ""),
                    "bb_2": row.get("bb_2", ""),
                    "date": row.get("date", ""),
                }
                if all(v == "" for v in data.values()):
                    continue
                result.append(data)
        return result
    except Exception as e:
        logging.error(f"解析本地csv数据至表格失败,csv文件 {path} 格式报错 {e}")
        return []

# 检查网络
def check_network() -> bool:
    url = "https://datachart.500.com/dlt/history/newinc/history.php?start=25057&end=25086"
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(url)
            if response.status_code == 200:
                logging.info(f"{url} 检测成功，状态码: {response.status_code}")
                return True 
            else:
                logging.error(f"{url} 检测失败，状态码: {response.status_code}")
                return False
    except Exception as e:
        logging.error(f"{url} 检测失败，异常: {e}")
        return False

def check_input_in_range(
        value: str,
        min_val: float,
        max_val: float,
        integer: bool = False
    ) -> Tuple[bool, Optional[Union[int, float]]]:
        value = value.strip()
        try:
            if integer:
                if not value.isdigit():
                    return False, None
                num = int(value)
            else:
                num = float(value)
        except ValueError:
            return False, None

        if min_val <= num <= max_val:
            return True, num
        else:
            return False, None

def is_valid_json(json_str: str) -> bool:
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False
