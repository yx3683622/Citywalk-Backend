# -*- coding: utf-8 -*-


import os
import re
import json
import math
import time
import logging
import requests
import pandas as pd
from collections import deque
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rapidfuzz import fuzz
from openai import OpenAI
from requests.exceptions import Timeout

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle



#  基础日志 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

#  Flask & CORS 
app = Flask(__name__)
app = Flask(__name__)
CORS(app, resources={
    r"/plan_route": {"origins": "*"},
    r"/chat_route": {"origins": "*"},
    r"/profile": {"origins": "*"}
}, supports_credentials=True)
#  配置 
CONFIG = {
    "LLM_API_URL": "https://integrate.api.nvidia.com/v1",
    "LLM_API_KEY": "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", 
    "LLM_MODEL": "deepseek-ai/deepseek-r1",
    "GAODE_API_KEY": "53a79ad00fd12cd20358c177df74384c", 
    "ALLOWED_TYPES": ["自然休闲型", "历史文化型", "美食体验型", "艺术潮流型", "社交娱乐型"],
    "LLM_MAX_RETRIES": 5,
    "LLM_RETRY_DELAYS": [2, 4, 8, 16, 32],
    "ROUTE_MIN_DISTANCE": 2000,
    "ROUTE_MAX_DISTANCE": 10000,
    "DEFAULT_DISTANCE_BASE": 5000,
    "DEFAULT_TIME_LIMIT": 240,
    "DEFAULT_STAY_TIME": "25分钟",
    "WALKING_SPEED": 80,
    "OPTIMAL_ROUTE_MAX_RETRIES": 3,
    "POI_COUNT_RANGE": [3, 6]
}

cluster_names = ["自然休闲型","历史文化型","艺术潮流型","美食体验型","社交娱乐型"]
sub_options_dict = {
    "nature": ["公园漫步","湖边休闲","城市绿道"],
    "history": ["历史建筑","博物馆","古街区"],
    "art": ["艺术展览","潮流街区","创意市集"],
    "food": ["美食街","餐饮探索","特色小吃"],
    "social": ["聚会","朋友小聚","热门景点拍照"]
}

# 加载模型 
with open(r"scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(r"kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open(r"gmm.pkl", "rb") as f:
    gmm = pickle.load(f)
with open(r"X_columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

# 初始化 LLM 
client = OpenAI(base_url=CONFIG["LLM_API_URL"], api_key=CONFIG["LLM_API_KEY"])

# 工具函数
def haversine(lon1, lat1, lon2, lat2):
    """返回两点间直线距离（米）"""
    R = 6371000
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def sharp_softmax(distances, temperature=0.2, threshold=0.05):
    inv_dist = 1 / (distances + 1e-6)
    inv_dist = inv_dist ** (1 / temperature)
    probs = inv_dist / inv_dist.sum(axis=1, keepdims=True)
    probs[probs < threshold] = 0
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs

def sharp_gmm_probs(probs, temperature=0.5, threshold=0.05):
    probs = probs ** (1 / temperature)
    probs = probs / probs.sum(axis=1, keepdims=True)
    probs[probs < threshold] = 0
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs

# 高德 POI 搜索
def poi_match(query):
    logger.info(f"高德API搜索: {query}")
    try:
        resp = requests.get(
            "https://restapi.amap.com/v5/place/text",
            params={
                "key": CONFIG["GAODE_API_KEY"],
                "keywords": query,
                "region": "上海",
                "city_limit": "true",
                "show_fields": "name,address,location"
            },
            timeout=8
        )
        if resp.status_code == 200 and resp.json().get('status') == '1':
            pois = resp.json().get('pois', [])
            if pois:
                best_gaode = max(pois, key=lambda p: fuzz.ratio(query, p.get('name','')))
                lon, lat = map(float, best_gaode['location'].split(','))
                if 120.8 <= lon <= 122.2 and 30.7 <= lat <= 31.8:
                    logger.info(f"匹配成功: {query} -> {best_gaode['name']}")
                    return {
                        'name': best_gaode['name'],
                        'location': [lon, lat],
                        'address': best_gaode.get('address', '')
                    }
    except Exception as e:
        logger.error(f"高德API失败: {str(e)}")
    return None

# 贪心 TSP
def tsp_optimize(points, target_dist):
    path = [0]
    unvisited = set(range(1, len(points)))
    # 第一步：基础最近邻排序
    while unvisited:
        last = path[-1]
        next_point = min(unvisited,
                         key=lambda x: haversine(points[last][0], points[last][1], points[x][0], points[x][1]))
        path.append(next_point)
        unvisited.remove(next_point)
    
    # 第二步：计算当前直线距离，若偏离目标则调整首尾POI
    current_dist = 0
    for i in range(len(path)-1):
        current_dist += haversine(points[path[i]][0], points[path[i]][1], points[path[i+1]][0], points[path[i+1]][1])
    
    # 距离过短调整
    if current_dist < CONFIG["ROUTE_MIN_DISTANCE"]:
        if unvisited:
            farthest_idx = max(unvisited, key=lambda x: haversine(points[path[0]][0], points[path[0]][1], points[x][0], points[x][1]))
            path[-1] = farthest_idx
        else:
            path = path[::-1]  # 反转顺序延长路径
    # 距离过长调整
    elif current_dist > CONFIG["ROUTE_MAX_DISTANCE"]:
        if unvisited:
            nearest_idx = min(unvisited, key=lambda x: haversine(points[path[0]][0], points[path[0]][1], points[x][0], points[x][1]))
            path[-1] = nearest_idx
    
    return path

# 高德 walking 路线
def get_gaode_route(points):
    DIR_DELTA = {
        '东': (0.00015, 0), '西': (-0.00015, 0),
        '南': (0, -0.00013), '北': (0, 0.00013),
        '右': (0.0001, 0), '左': (-0.0001, 0)
    }
    correction_factor = 1.2  # 步行时间修正系数
    distances = []
    durations = []
    segments = []
    full_path = []
    try:
        for i in range(len(points) - 1):
            origin = f"{points[i][0]},{points[i][1]}"
            destination = f"{points[i + 1][0]},{points[i + 1][1]}"
            retries = 0
            success = False
            while not success and retries < 3:
                try:
                    resp = requests.get(
                        "https://restapi.amap.com/v3/direction/walking",
                        params={
                            "key": CONFIG["GAODE_API_KEY"],
                            "origin": origin,
                            "destination": destination,
                            "extensions": "all"
                        },
                        timeout=8
                    )
                    data = resp.json()
                    if data.get('status') == '1' and data.get('route'):
                        path = data['route']['paths'][0]
                        distances.append(float(path['distance']))
                        durations.append(float(path['duration']) / 60 * correction_factor)  
                        
                        # 解析路径点（压缩冗余）
                        segment_points = []
                        for step in path.get('steps', []):
                            if polyline := step.get('polyline', ''):
                                for pair in polyline.split(';'):
                                    try:
                                        lon, lat = map(float, pair.split(','))
                                        segment_points.append((lon, lat))
                                        full_path.append((lon, lat))
                                    except:
                                        logger.warning(f"无效坐标: {pair}")
                            else:
                                start = list(map(float, step.get('start_location', '').split(','))) if step.get('start_location') else None
                                end = list(map(float, step.get('end_location', '').split(','))) if step.get('end_location') else start
                                if start and end:
                                    segment_points.extend([(start[0], start[1]), (end[0], end[1])])
                                    full_path.extend([(start[0], start[1]), (end[0], end[1])])
                                    if (instruction := step.get('instruction')) and (match := re.search(r'向([东南西北右左])', instruction)):
                                        dx, dy = DIR_DELTA.get(match.group(1), (0, 0))
                                        segment_points.append((end[0] + dx, end[1] + dy))
                                        full_path.append((end[0] + dx, end[1] + dy))
                        
                        # 压缩路径点
                        compressed_segment = []
                        last_point = None
                        for point in segment_points:
                            if last_point is None or haversine(last_point[0], last_point[1], point[0], point[1]) >= 0.01:
                                compressed_segment.append(point)
                                last_point = point
                        segments.append({
                            "points": compressed_segment,
                            "distance": float(path['distance']),
                            "duration": float(path['duration']) / 60 * correction_factor
                        })
                        success = True
                except Exception as e:
                    logger.warning(f"高德API重试: {str(e)}, 第{retries + 1}次")
                    retries += 1
                    time.sleep(0.5 * retries)
        return {
            "path": {"segments": segments, "full": full_path},
            "distance": round(sum(distances), 2),  # 实际步行距离（米）
            "duration": round(sum(durations), 2)   # 实际步行时间（分钟）
        }
    except Exception as e:
        logger.error(f"路线规划失败: {str(e)}")
        return {"path": {"segments": [], "full": []}, "distance": 0, "duration": 0}

#  约束解析 
def parse_constraints(user_input, user_profile):
    # 内部使用的完整约束
    full_constraints = {
        "distance_desc": "5km（推荐基准，实际路线需在2km-10km之间）",
        "distance_min": CONFIG["ROUTE_MIN_DISTANCE"],
        "distance_max": CONFIG["ROUTE_MAX_DISTANCE"],
        "distance_base": CONFIG["DEFAULT_DISTANCE_BASE"],
        "time": "3h以内",
        "time_limit_min": CONFIG["DEFAULT_TIME_LIMIT"],
        "location": "",
        "budget": "",
        "type": []
    }

    # 解析距离约束
    dist_match = re.search(r'(\d+)(km|米)', user_input)
    if dist_match:
        dist_val = int(dist_match.group(1))
        dist_unit = dist_match.group(2)
        if dist_unit == "km":
            user_dist_m = dist_val * 1000
            full_constraints["distance_desc"] = f"{dist_val}km（实际路线需在2km-10km之间）"
        else:
            user_dist_m = dist_val
            full_constraints["distance_desc"] = f"{dist_val}米（实际路线需在2000米-10000米之间）"
        
        # 强制落在区间内
        if user_dist_m < CONFIG["ROUTE_MIN_DISTANCE"]:
            full_constraints["distance_base"] = CONFIG["ROUTE_MIN_DISTANCE"]
        elif user_dist_m > CONFIG["ROUTE_MAX_DISTANCE"]:
            full_constraints["distance_base"] = CONFIG["ROUTE_MAX_DISTANCE"]
        else:
            full_constraints["distance_base"] = user_dist_m

    # 解析时间约束
    time_match = re.search(r'(\d+)(h|小时|分钟)', user_input)
    if time_match:
        time_val = int(time_match.group(1))
        time_unit = time_match.group(2)
        if time_unit in ["h", "小时"]:
            full_constraints["time_limit_min"] = min(time_val * 60, 360)  # 最大6小时
            full_constraints["time"] = f"{time_val}h以内"
        else:
            full_constraints["time_limit_min"] = min(time_val, 360)
            full_constraints["time"] = f"{time_val}分钟以内"

    # 解析POI类型
    for poi_type in CONFIG["ALLOWED_TYPES"]:
        if poi_type in user_input or poi_type in user_profile:
            if poi_type not in full_constraints["type"]:
                full_constraints["type"].append(poi_type)
    if not full_constraints["type"]:
        full_constraints["type"] = ["自然休闲型"]  # 默认类型

    # 生成对外展示的精简约束（仅保留需要的字段）
    display_constraints = {
        "time": full_constraints["time"],
        "distance": full_constraints["distance_desc"],
        "location": full_constraints["location"],
        "budget": full_constraints["budget"],
        "type": full_constraints["type"]
    }

    logger.info(f"解析后约束: {display_constraints}")
    return full_constraints, display_constraints

#  路线区间验证/调整
def calculate_route_linear_distance(pois):
    """计算按给定顺序 pois 的直线总距离（米），pois 每项包含 longitude, latitude"""
    if len(pois) < 2:
        return 0
    total = 0.0
    for i in range(len(pois)-1):
        lon1, lat1 = pois[i]['longitude'], pois[i]['latitude']
        lon2, lat2 = pois[i+1]['longitude'], pois[i+1]['latitude']
        total += haversine(lon1, lat1, lon2, lat2)
    return round(total, 2)

def is_route_in_range(pois, full_constraints):
    poi_count = len(pois)
    # 验证POI数量
    if not (CONFIG["POI_COUNT_RANGE"][0] <= poi_count <= CONFIG["POI_COUNT_RANGE"][1]):
        msg = f"POI数量不合理: {poi_count}个（需{CONFIG['POI_COUNT_RANGE'][0]}-{CONFIG['POI_COUNT_RANGE'][1]}个）"
        return False, msg
    
    # 验证直线距离
    linear_dist = calculate_route_linear_distance(pois)
    if linear_dist < full_constraints["distance_min"] * 0.8:
        msg = f"直线距离过短: {linear_dist:.0f}米（需≥{full_constraints['distance_min']*0.8:.0f}米）"
        return False, msg
    if linear_dist > full_constraints["distance_max"] * 1.1:
        msg = f"直线距离过长: {linear_dist:.0f}米（需≤{full_constraints['distance_max']*1.1:.0f}米）"
        return False, msg
    
    return True, f"初步验证通过（POI{poi_count}个，直线距离{linear_dist:.0f}米）"

def adjust_route_to_range(pois, full_constraints):
    # 优化POI顺序
    points = [(p['longitude'], p['latitude']) for p in pois]
    optimized_indices = tsp_optimize(points, full_constraints["distance_base"])
    adjusted_pois = [pois[i] for i in optimized_indices]
    
    # 获取实际步行距离
    route_info = get_gaode_route([(p['longitude'], p['latitude']) for p in adjusted_pois])
    actual_dist = route_info["distance"]
    
    # 增强短距离修复逻辑
    retry_count = 0
    max_retries = 3
    while retry_count < max_retries and actual_dist < full_constraints["distance_min"]:
        logger.info(f"步行距离{actual_dist:.0f}米<3km，尝试补充POI延长路线")
        # 计算现有POI的地理中心
        avg_lon = sum(p['longitude'] for p in adjusted_pois) / len(adjusted_pois)
        avg_lat = sum(p['latitude'] for p in adjusted_pois) / len(adjusted_pois)
        # 搜索中心周边2km内的同类型POI
        main_type = adjusted_pois[0]['type'].split(',')[0] if adjusted_pois else "自然休闲型"
        new_poi = poi_match(f"{main_type} 上海 {avg_lon:.4f},{avg_lat:.4f} 2km内")
        
        if new_poi and new_poi["name"] not in [p["name"] for p in adjusted_pois]:
            # 转换新POI格式为我们需要的字段
            formatted_poi = {
                "name": new_poi["name"],
                "longitude": new_poi["location"][0],
                "latitude": new_poi["location"][1],
                "type": main_type,
                "reason": f"1. 补充同类型{main_type}POI；2. 延长路线距离至合理区间",
                "time": CONFIG["DEFAULT_STAY_TIME"]
            }
            adjusted_pois.append(formatted_poi)
            # 重新优化顺序并计算距离
            points = [(p['longitude'], p['latitude']) for p in adjusted_pois]
            optimized_indices = tsp_optimize(points, full_constraints["distance_base"])
            adjusted_pois = [adjusted_pois[i] for i in optimized_indices]
            route_info = get_gaode_route([(p['longitude'], p['latitude']) for p in adjusted_pois])
            actual_dist = route_info["distance"]
        
        retry_count += 1
    
    # 处理过长距离
    while retry_count < max_retries and actual_dist > full_constraints["distance_max"]:
        logger.info(f"步行距离{actual_dist:.0f}米>10km，尝试减少POI缩短路线")
        if len(adjusted_pois) > CONFIG["POI_COUNT_RANGE"][0]:
            # 移除离起点最远的POI
            farthest_idx = max(range(len(adjusted_pois)), key=lambda i: haversine(
                adjusted_pois[0]['longitude'], adjusted_pois[0]['latitude'],
                adjusted_pois[i]['longitude'], adjusted_pois[i]['latitude']
            ))
            adjusted_pois.pop(farthest_idx)
            # 重新计算
            points = [(p['longitude'], p['latitude']) for p in adjusted_pois]
            optimized_indices = tsp_optimize(points, full_constraints["distance_min"])
            adjusted_pois = [adjusted_pois[i] for i in optimized_indices]
            route_info = get_gaode_route([(p['longitude'], p['latitude']) for p in adjusted_pois])
            actual_dist = route_info["distance"]
        retry_count += 1
    
    # 验证时间约束
    total_stay_min = 0
    for poi in adjusted_pois:
        stay_time = poi.get("time", CONFIG["DEFAULT_STAY_TIME"])
        stay_match = re.search(r'(\d+)', stay_time)
        stay_val = int(stay_match.group(1)) if stay_match else 25
        if "小时" in stay_time or "h" in stay_time:
            stay_val *= 60
        total_stay_min += stay_val
    
    total_time_min = route_info["duration"] + total_stay_min
    if total_time_min > full_constraints["time_limit_min"]:
        # 按比例压缩停留时间（最小15分钟）
        excess_time = total_time_min - full_constraints["time_limit_min"]
        compression_ratio = 1 - (excess_time / total_stay_min)
        compression_ratio = max(compression_ratio, 0.6)  # 最多压缩40%
        
        for poi in adjusted_pois:
            stay_time = poi.get("time", CONFIG["DEFAULT_STAY_TIME"])
            stay_match = re.search(r'(\d+)', stay_time)
            stay_val = int(stay_match.group(1)) if stay_match else 25
            is_hour = "小时" in stay_time or "h" in stay_time
            if is_hour:
                stay_val *= 60
            
            new_stay_val = max(round(stay_val * compression_ratio), 15)
            if new_stay_val >= 60:
                poi["time"] = f"{new_stay_val // 60}小时{new_stay_val % 60}分钟"
            else:
                poi["time"] = f"{new_stay_val}分钟"
        
        # 重新计算总时间
        total_stay_min = sum(
            int(re.search(r'(\d+)', p["time"]).group(1)) * (60 if "小时" in p["time"] or "h" in p["time"] else 1)
            for p in adjusted_pois
        )
        total_time_min = route_info["duration"] + total_stay_min

    # 最终结果封装
    route_info["total_stay_min"] = total_stay_min
    route_info["total_time_min"] = total_time_min
    route_info["actual_distance_range"] = f"{actual_dist:.0f}米（符合3000-10000米区间）" if (full_constraints["distance_min"] <= actual_dist <= full_constraints["distance_max"]) else f"{actual_dist:.0f}米（已尽力调整）"
    return adjusted_pois, route_info

# LLM 调用与 JSON 提取 
def extract_json_from_response(response_text):
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            raise ValueError("未找到有效JSON")
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}，原始文本前500字符: {response_text[:500]}")
        raise
    except Exception as e:
        logger.error(f"提取JSON失败: {str(e)}")
        raise

def extract_deepseek_thoughts(text):
    """提取 <think> 思考过程 和 最终回答部分"""
    think = ""
    answer = text
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if m:
        think = m.group(1).strip()
        answer = text.replace(m.group(0), "").strip()
    return think, answer



def call_llm_for_dataset(user_profile, user_input):
    # 解析约束（生成完整约束和展示约束）
    full_constraints, display_constraints = parse_constraints(user_input, user_profile)
    
    # 构造System Prompt，强调严格的输出格式
    system_prompt = f"""你是上海POI路线规划专家，必须严格遵守以下约束：
### 1. 路线距离约束（核心）
- 区间要求：实际步行总距离必须在{full_constraints['distance_min']/1000}km-{full_constraints['distance_max']/1000}km之间
- 默认基准：用户无明确要求时，推荐以{full_constraints['distance_base']/1000}km为核心
- 区域要求：POI需集中在上海同一或相邻区域，避免跨区导致距离失控

### 2. 时间约束
- 总时间 = 步行时间 + 所有POI停留时间 ≤ {full_constraints['time_limit_min']}分钟（{full_constraints['time']}）
- 停留时间：单POI15-45分钟（自然/历史类30-45分钟，美食/潮流类15-30分钟）

### 3. POI要求
- 数量：{CONFIG['POI_COUNT_RANGE'][0]}-{CONFIG['POI_COUNT_RANGE'][1]}个（上海真实地点）
- 类型：从{', '.join(CONFIG['ALLOWED_TYPES'])}中选，优先{full_constraints['type']}
- 经纬度：精确到小数点后4位（如121.4875，31.2272），必须为数字类型（无引号）

### 输出格式（严格JSON，仅包含指定字段，禁止添加任何额外字段）
{{
  "constraints": {{
    "time": "{display_constraints['time']}",
    "distance": "{display_constraints['distance']}",
    "location": "{display_constraints['location']}",
    "budget": "{display_constraints['budget']}",
    "type": {json.dumps(display_constraints['type'])}
  }},
  "pois": [
    {{
      "name": "上海真实POI名称",
      "longitude": 0.0000,  # 经度，数字类型，4位小数
      "latitude": 0.0000,   # 纬度，数字类型，4位小数
      "type": "",  # 类型，可多个（逗号分隔）
      "reason": "1. 结合用户画像{user_profile[:30]}...；2. 符合{display_constraints['type'][0]}类型；3. 助力路线距离合规",
      "time": "停留时间（如'10分钟','20分钟'等）"
    }}
  ]
}}

特别注意：
- 输出只能是上述格式的JSON，不能有任何多余文字
- 严格按照字段列表输出，不得添加额外字段（如location、address等）
- longitude和latitude必须是数字类型，不能带引号
"""

    user_prompt = f"用户画像：{user_profile}\n用户需求：{user_input}\n请按3-10km区间和{full_constraints['distance_base']/1000}km基准生成路线"

    # LLM调用+重试
    for attempt in range(CONFIG["LLM_MAX_RETRIES"]):
        try:
            completion = client.chat.completions.create(
                model=CONFIG["LLM_MODEL"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.55,
                max_tokens=2500,
                timeout=120
            )

            raw_reply = completion.choices[0].message.content

            # ✅ 提取 DeepSeek 思考链
            think, answer = extract_deepseek_thoughts(raw_reply)
            if think:
                logger.info("[LLM思考链]:\n" + think)
            

            json_output = extract_json_from_response(answer)
            pois = json_output["pois"]

            # 验证并清理POI字段
            for poi in pois:
                # 验证并转换经纬度为数字
                try:
                    poi["longitude"] = round(float(poi["longitude"]), 4)
                    poi["latitude"] = round(float(poi["latitude"]), 4)
                except:
                    raise ValueError(f"POI「{poi.get('name')}」经纬度错误（需为数字，如121.4875）")
                
                # 确保停留时间格式正确
                if not re.search(r'(\d+)(小时|分钟|h)', poi.get("time", "")):
                    poi["time"] = CONFIG["DEFAULT_STAY_TIME"]
                
                # 移除可能存在的额外字段
                allowed_keys = ["name", "longitude", "latitude", "type", "reason", "time"]
                for key in list(poi.keys()):
                    if key not in allowed_keys:
                        del poi[key]

            # 预校验路线是否符合区间
            is_pre_valid, pre_msg = is_route_in_range(pois, full_constraints)
            if not is_pre_valid:
                raise ValueError(f"路线预校验失败：{pre_msg}")

            # 调整路线至合理区间
            adjusted_pois, route_info = adjust_route_to_range(pois, full_constraints)
            actual_dist = route_info["distance"]

            # 最终验证区间和时间
            if not (full_constraints["distance_min"] - 500 <= actual_dist <= full_constraints["distance_max"] + 500):
                raise ValueError(f"路线距离仍超出3-10km区间：{actual_dist:.0f}米（已尽力调整）")
            if route_info["total_time_min"] > full_constraints["time_limit_min"] + 10:
                raise ValueError(f"总时间超约束：{route_info['total_time_min']:.0f}分钟（约束{full_constraints['time_limit_min']}分钟）")

            # 使用精简的约束信息
            json_output["constraints"] = display_constraints
            json_output["pois"] = adjusted_pois            
            full_path = route_info.get("path", {}).get("full", []) or []

            try:
                json_output["path"] = [[float(pt[0]), float(pt[1])] for pt in full_path]
            except Exception:       
                json_output["path"] = []

            logger.info(f"LLM调用成功（第{attempt+1}次）：路线{actual_dist:.0f}米（3-10km区间内），总时间{route_info['total_time_min']:.0f}分钟")
            return (
                json_output,
                actual_dist,
                route_info["duration"],
                [p["time"] for p in adjusted_pois],
            )

        except Timeout:
            logger.warning(f"LLM超时（第{attempt+1}次），{CONFIG['LLM_RETRY_DELAYS'][attempt]}秒后重试")
            if attempt < CONFIG["LLM_MAX_RETRIES"] - 1:
                time.sleep(CONFIG["LLM_RETRY_DELAYS"][attempt])
            else:
                raise Exception(f"LLM调用超时（重试{CONFIG['LLM_MAX_RETRIES']}次）")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM调用失败（第{attempt+1}次）：{error_msg}")
            if attempt < CONFIG["LLM_MAX_RETRIES"] - 1:
                user_prompt = f"用户画像：{user_profile}\n用户需求：{user_input}\n上一次失败原因：{error_msg}\n请修正：1. 确保POI步行距离在3-10km；2. 总时间≤{full_constraints['time_limit_min']}分钟；3. POI数量{CONFIG['POI_COUNT_RANGE'][0]}-{CONFIG['POI_COUNT_RANGE'][1]}个；4. 只保留指定字段，不要添加额外字段"
                time.sleep(CONFIG["LLM_RETRY_DELAYS"][attempt])
            else:
                raise Exception(f"LLM调用失败（重试{CONFIG['LLM_MAX_RETRIES']}次）：{error_msg}")

def summarize_route(pois, dist_m, time_min):
    names = " → ".join([p["name"] for p in pois])
    return f"为你规划了约{dist_m/1000:.1f}公里的Citywalk路线，预计{int(time_min)}分钟：{names}"

# 最终约束验证
def validate_constraints_final(full_constraints, actual_dist, actual_duration, poi_stay_times):
    # 距离验证（允许±500米误差）
    distance_valid = (full_constraints["distance_min"] - 500) <= actual_dist <= (full_constraints["distance_max"] + 500)
    
    # 时间验证
    total_stay_min = sum(
        int(re.search(r'(\d+)', t).group(1)) * (60 if "小时" in t or "h" in t else 1)
        for t in poi_stay_times
    )
    total_time_min = actual_duration + total_stay_min
    time_valid = total_time_min <= (full_constraints["time_limit_min"] + 10)  # 允许10分钟误差

    logger.info(f"最终验证：距离{actual_dist:.0f}米（3-10km）→ {'符合' if distance_valid else '接近'}；时间{total_time_min:.0f}分钟→ {'符合' if time_valid else '超期'}")
    return distance_valid, time_valid

# 会话管理
SESSIONS = {}

def new_session():
    """新建 session，返回 session_id（时间戳方式）"""
    sid = str(int(time.time() * 1000))
    SESSIONS[sid] = {"created_at": time.time(), "messages": deque(maxlen=50), "last_route": None}
    logger.info(f"[SESSION] 新建 session {sid}")
    return sid

def get_session(session_id):
    s = SESSIONS.get(session_id)
    if not s:
        return None
    if (time.time() - s["created_at"]) > CONFIG.get("SESSION_TTL_SEC", 3600*6):
        SESSIONS.pop(session_id, None)
        return None
    return s

def record_msg(session, role, content):
    session["messages"].append({"role": role, "content": content, "ts": int(time.time())})

def pretty_log_route(endpoint, pois, dist, duration,user_input):
    logger.info("\n" + "="*60)
    logger.info(f"CityWalk路径规划 | 接口: {endpoint}")
    logger.info("-"*60)
    logger.info(f"用户输入: {user_input}")
    logger.info(f"路线距离: {dist/1000:.2f} km")
    logger.info(f"⏱预计总时长: {int(duration)} 分钟")
    logger.info(f"POI数量: {len(pois)}")
    logger.info(" ")

    logger.info("POI详情：")
    for i,p in enumerate(pois,1):
        logger.info(f"{i}) {p['name']} ({p['longitude']}, {p['latitude']})")
        logger.info(f"   类型: {p['type']}")
        logger.info(f"   理由: {p['reason']}")
        logger.info(f"   停留: {p['time']}")
    logger.info(" ")

    seq = " → ".join([p["name"] for p in pois])
    logger.info(f"路线顺序:\n{seq}")
    logger.info("="*60 + "\n")


def pretty_log_profile(result):
    logger.info("\n" + "="*60)
    logger.info("用户画像计算完成")
    logger.info("-"*60)
    primary = result["tags"]["primary"]
    logger.info(f"主类型: {primary}")

# Flask 路由
@app.route("/plan_route", methods=["GET"])
def plan_route():
    try:
        user_input = (request.args.get("user_input") or "").strip()
        user_tags = (request.args.get("user_tags") or "").strip()
        session_id = (request.args.get("session_id") or "").strip()

        if not user_input:
            return jsonify({"error": "请输入有效内容"}), 400

        session = get_session(session_id) if session_id else None
        if not session:
            session_id = new_session()
            session = get_session(session_id)

        record_msg(session, "user", user_input)

        output, actual_dist, actual_dur, poi_times = call_llm_for_dataset(user_tags, user_input)
        session["last_route"] = output

        # ✅ 展示日志
        pretty_log_route("/plan_route", output["pois"], actual_dist, actual_dur,user_input)

        return jsonify({"success": True,"session_id": session_id,"data": output})
    except Exception as e:
        logger.error(e)
        return jsonify({"error": "服务器内部错误"}), 500


@app.route("/chat_route", methods=["POST"])
def chat_route():
    try:
        body = request.get_json(force=True, silent=True) or {}
        session_id = (body.get("session_id") or "").strip()
        user_turn = (body.get("user_input") or "").strip()
        user_tags = (body.get("user_tags") or "").strip()

        if not session_id: return jsonify({"error": "缺少session_id"}), 400
        if not user_turn: return jsonify({"error": "请输入内容"}), 400

        session = get_session(session_id)
        if not session or not session.get("last_route"):
            return jsonify({"error": "请先调用 plan_route"}), 400

        record_msg(session,"user",user_turn)

        output, actual_dist, actual_dur, poi_times = call_llm_for_dataset(user_tags, user_turn)
        session["last_route"] = output

        # ✅ 展示日志
        pretty_log_route("/chat_route", output["pois"], actual_dist, actual_dur,user_turn )

        return jsonify({"success": True,"session_id": session_id,"data": output})
    except Exception as e:
        logger.error(e)
        return jsonify({"error": "服务器错误"}), 500

@app.route("/profile", methods=["POST"])
def profile_api():
    data = request.get_json(force=True)
    answers = data.get("answers", {})

    # 处理 answers
    q2_selected = answers.get("q2", [])
    sub_scores = {}
    for val in q2_selected:
        sid = f"sub_{val}"
        sub_scores[val] = answers.get(sid, [])

    # 规则概率（子类打分） 
    if q2_selected:
        weights = {}
        for val in q2_selected:
            total = sum(sub_scores.get(val, [1]))
            weights[val] = total if total > 0 else 1
        total_w = sum(weights.values())
        probs = {c: (weights.get(cmap, 0) / total_w * 100 if total_w > 0 else 0) for cmap, c in {
            "nature": "自然休闲型",
            "history": "历史文化型",
            "art": "艺术潮流型",
            "food": "美食体验型",
            "social": "社交娱乐型"}.items()}
    else:
        # 如果没答子题，走模型预测
        df_dummy = pd.DataFrame(np.zeros((1, len(train_columns))), columns=train_columns)
        X_scaled = scaler.transform(df_dummy)
        k_probs = sharp_softmax(kmeans.transform(X_scaled))
        g_probs = sharp_gmm_probs(gmm.predict_proba(X_scaled))
        final_probs = [(k_probs[0, i] + g_probs[0, i]) / 2 for i in range(len(cluster_names))]
        total = sum(final_probs)
        probs = {cluster_names[i]: round(final_probs[i] / total * 100, 1) for i in range(len(cluster_names))}

    # 结果 
    final_category = max(probs, key=probs.get)
    result = {
        "success": True,
        "tags": {"primary": final_category,"all":[final_category]},
        "probs": probs
    }

    # ✅ 新版漂亮日志
    pretty_log_profile(result)

    return jsonify(result)



@app.route("/")
def index():
    return "Citywalk 后端已启动。调用 /plan_route 或 /chat_route"

# 启动 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
