import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import os
import time
import warnings
import traceback
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Callable

warnings.filterwarnings('ignore', category=UserWarning)
class LotteryPredictor:
    
    def __init__(self,
                 lookback_periods: int = 15,
                 model_type: str = "lightgbm",
                 verbose: bool = True,
                 feature_engineering: bool = True,
                 test_size: float = 0.2,
                 model: Optional[Any] = None,
                 render: Optional[Any] = None,
                 file_path: str = ''
                 ):
        
        self.lookback_periods = lookback_periods
        self.model_type = model_type
        self.verbose = verbose
        self.feature_engineering = feature_engineering
        self.test_size = test_size
        self.model = model
        self.render = render
        self.file_path = file_path
        
        # 模型存储
        self.rb_model = None
        self.rb_encoder = None
        self.bb_model = None
        self.bb_encoder = None
        self.features = None
        self.targets = None
        self.df = None 
    
    def render_method(self, message: str):
        self.render(message)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """从CSV文件加载数据并处理"""
        self.render_method(f"正在加载数据文件: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        start_time = time.time()
        df = pd.read_csv(file_path)
        load_time = time.time() - start_time
        
        self.render_method(f"数据加载完成，耗时 {load_time:.2f} 秒")
        self.render_method(f"数据维度: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 删除可能存在的空行
        original_count = len(df)
        df = df.dropna()
        cleaned_count = len(df)
        if cleaned_count < original_count:
            self.render_method(f"已删除 {original_count - cleaned_count} 行空数据")
        
        # 确保号码列是整数类型
        number_cols = ['rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5', 'bb_1', 'bb_2']
        for col in number_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # 反转数据使时间升序（最新的在最后）
        df = df.iloc[::-1].reset_index(drop=True)
        
        # 日期范围
        if 'date' in df.columns:
            self.render_method(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
        self.df = df  # 保存数据供其他方法使用
        return df
    
    def create_basic_features(self, df: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """创建基本特征数据集"""
        self.render_method(f"开始创建基本特征数据集 (使用过去 {lookback} 期数据)...")
        start_time = time.time()
        
        all_columns = ['rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5', 'bb_1', 'bb_2']
        
        features, rb_targets, bb_targets = [], [], []
        
        for i in range(lookback, len(df)):
            # 基础特征：过去lookback期的所有号码
            past_numbers = []
            for j in range(i - lookback, i):
                # 确保只使用存在的列
                row_values = [df.iloc[j][col] for col in all_columns if col in df.columns]
                past_numbers.extend(row_values)
            
            # 标签：当前期的号码
            if 'rb_1' in df.columns:
                rb_targets.append(df.iloc[i][['rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5']].values)
            if 'bb_1' in df.columns:
                bb_targets.append(df.iloc[i][['bb_1', 'bb_2']].values)
            features.append(past_numbers)
        
        process_time = time.time() - start_time
        self.render_method(f"特征创建完成，耗时 {process_time:.2f} 秒")
        self.render_method(f"特征维度: {len(features)} 个样本, {len(features[0])} 个特征")
        
        return np.array(features), np.array(rb_targets), np.array(bb_targets)
    
    def create_advanced_features(self, df: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """创建增强特征数据集"""
        self.render_method(f"开始创建增强特征数据集 (使用过去 {lookback} 期数据)...")
        start_time = time.time()
        
        all_columns = ['rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5', 'bb_1', 'bb_2']
        
        features, rb_targets, bb_targets = [], [], []
        
        # 创建号码范围
        rb_range = list(range(1, 36))  # 前区号码范围(1-35)
        bb_range = list(range(1, 13))  # 后区号码范围(1-12)
        
        total_iterations = len(df) - lookback
        last_log_time = time.time()
        
        for i in range(lookback, len(df)):
            # 基础特征：过去lookback期的所有号码
            past_numbers = []
            for j in range(i - lookback, i):
                past_numbers.extend(df.iloc[j][all_columns].values)
            
            # 1. 添加频率特征
            freq_features = []
            # 前区号码频率
            for num in rb_range:
                count = sum(1 for j in range(i - lookback, i) if num in df.iloc[j][['rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5']].values)
                freq_features.append(count)
            
            # 后区号码频率
            for num in bb_range:
                count = sum(1 for j in range(i - lookback, i) if num in df.iloc[j][['bb_1', 'bb_2']].values)
                freq_features.append(count)
            
            # 2. 添加遗漏值特征
            missing_features = []
            # 前区号码遗漏值
            for num in rb_range:
                missing = 0
                for j in range(i-1, max(i-10, -1), -1):  # 最多回溯10期
                    if num in df.iloc[j][['rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5']].values:
                        break
                    missing += 1
                missing_features.append(missing)
            
            # 后区号码遗漏值
            for num in bb_range:
                missing = 0
                for j in range(i-1, max(i-10, -1), -1):  # 最多回溯10期
                    if num in df.iloc[j][['bb_1', 'bb_2']].values:
                        break
                    missing += 1
                missing_features.append(missing)
            
            # 3. 添加和值特征
            sum_features = []
            for j in range(i - lookback, i):
                rb_sum = sum(df.iloc[j][['rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5']].values)
                bb_sum = sum(df.iloc[j][['bb_1', 'bb_2']].values)
                sum_features.append(rb_sum)
                sum_features.append(bb_sum)
            
            # 4. 添加奇偶比例特征
            parity_features = []
            for j in range(i - lookback, i):
                rb_odd = sum(1 for num in df.iloc[j][['rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5']].values if num % 2 == 1)
                bb_odd = sum(1 for num in df.iloc[j][['bb_1', 'bb_2']].values if num % 2 == 1)
                parity_features.append(rb_odd / 5.0)
                parity_features.append(bb_odd / 2.0)
            
            # 组合所有特征
            full_features = past_numbers + freq_features + missing_features + sum_features + parity_features
            
            # 标签：当前期的号码
            rb_targets.append(df.iloc[i][['rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5']].values)
            bb_targets.append(df.iloc[i][['bb_1', 'bb_2']].values)
            features.append(full_features)
            
            # 定期记录进度
            current_iter = i - lookback
            if current_iter % 100 == 0 or current_iter == total_iterations - 1:
                elapsed_time = time.time() - last_log_time
                self.render_method(f"特征工程进度: {current_iter+1}/{total_iterations} ({(current_iter+1)/total_iterations*100:.1f}%) - 最近100期耗时: {elapsed_time:.2f}秒")
                last_log_time = time.time()
        
        process_time = time.time() - start_time
        self.render_method(f"特征创建完成，耗时 {process_time:.2f} 秒")
        self.render_method(f"特征维度: {len(features)} 个样本, {len(features[0])} 个特征")
        
        return np.array(features), np.array(rb_targets), np.array(bb_targets)

    def train_model(self, X: np.ndarray, y: np.ndarray, target_type: str = "rb") -> Tuple[Any, LabelEncoder]:
        """训练模型"""
        log_prefix = f"[{'前区' if target_type == 'rb' else '后区'}模型]"
        self.render_method(f"{log_prefix} 开始训练")
        self.render_method(f"{log_prefix} 特征矩阵形状: {X.shape}")
        
        # 将多标签目标转换为单个标签
        self.render_method(f"{log_prefix} 组合标签...")
        if target_type == "rb":
            y_combined = [f"{a}_{b}_{c}_{d}_{e}" for a, b, c, d, e in y]
        else:
            y_combined = [f"{a}_{b}" for a, b in y]
        
        # 编码标签
        self.render_method(f"{log_prefix} 编码标签...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_combined)
        
        # 统计标签分布
        unique_labels = len(np.unique(y_encoded))
        self.render_method(f"{log_prefix} 标签编码完成，共 {unique_labels} 种组合")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=42
        )
        self.render_method(f"{log_prefix} 训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
        
        # 使用自定义模型构建器或默认构建器
        if self.model is not None:
            model = self.model
            self.render_method(f"{log_prefix} 使用自定义模型构建器创建模型")
        else:
            # 默认模型构建器
            if self.model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )
            else:  # 默认使用LightGBM
                model = LGBMClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    num_leaves=16,
                    min_child_samples=20,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            self.render_method(f"{log_prefix} 使用默认模型构建器创建模型")
        
        # 训练模型
        self.render_method(f"{log_prefix} 开始训练{self.model_type}模型...")
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        self.render_method(f"{log_prefix} 模型训练完成，耗时 {train_time:.2f} 秒")
        
        # 评估模型
        self.render_method(f"{log_prefix} 评估模型性能...")
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        self.render_method(f"{log_prefix} 训练集准确率: {train_acc:.4f}")
        
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        self.render_method(f"{log_prefix} 测试集准确率: {test_acc:.4f}")
        
        # 特征重要性
        try:
            importances = model.feature_importances_
            # 只显示非零重要性的特征
            non_zero_importances = importances[importances > 0]
            if len(non_zero_importances) > 0:
                top_features = np.argsort(importances)[::-1][:min(5, len(non_zero_importances))]
                self.render_method(f"{log_prefix} 最重要的{len(top_features)}个特征位置: {top_features}")
                self.render_method(f"{log_prefix} 最大特征重要性: {importances[top_features[0]]:.4f}")
            else:
                self.render_method(f"{log_prefix} 所有特征重要性均为零")
        except Exception as e:
            self.render_method(f"{log_prefix} 无法获取特征重要性: {str(e)}")
        
        return model, le
    
    def predict(self, last_features: np.ndarray, target_type: str = "rb") -> List[int]:
        """预测下一期号码"""
        log_prefix = f"[{'前区' if target_type == 'rb' else '后区'}预测]"
        self.render_method(f"{log_prefix} 开始预测")
        
        start_time = time.time()
        
        # 选择模型
        if target_type == "rb":
            model = self.rb_model
            le = self.rb_encoder
        else:
            model = self.bb_model
            le = self.bb_encoder
        
        if model is None or le is None:
            raise ValueError(f"{target_type}模型未训练")
        
        # 预测
        encoded_pred = model.predict(last_features)[0]
        pred_str = le.inverse_transform([encoded_pred])[0]
        
        # 解析预测结果
        preds = [int(x) for x in pred_str.split('_')]
        
        predict_time = time.time() - start_time
        self.render_method(f"{log_prefix} 预测完成，耗时 {predict_time:.4f} 秒")
        self.render_method(f"{log_prefix} 预测结果: {sorted(preds)}")
        
        return preds
    
    def start(self):
        rb, bb = self.train_and_predict()
        recent_results = self.get_recent_results(5)
        self.render_method("\n最近5期开奖结果:")
        self.render_method(str(recent_results))
        self.render_method(f"\n最终预测结果: 前区 {rb}, 后区 {bb}")

    def train_and_predict(self) -> Tuple[List[int], List[int]]:
        """完整流程：加载数据、训练模型、预测"""
        try:
            # 记录开始信息
            self.render_method("\n" + "="*60)
            self.render_method(f"开始预测 (模型类型: {self.model_type.upper()}, 历史期数: {self.lookback_periods})")
            self.render_method("="*60)
            
            total_start = time.time()
            
            # 加载数据
            df = self.load_data(self.file_path)
            
            # 创建特征
            if self.feature_engineering:
                X, y_rb, y_bb = self.create_advanced_features(df, self.lookback_periods)
            else:
                X, y_rb, y_bb = self.create_basic_features(df, self.lookback_periods)
            
            # 存储特征和目标
            self.features = X
            self.targets = (y_rb, y_bb)
            
            # 训练前区模型
            self.render_method("\n" + "="*60)
            self.render_method("训练前区模型")
            self.render_method("="*60)
            self.rb_model, self.rb_encoder = self.train_model(X, y_rb, "rb")
            
            # 训练后区模型
            self.render_method("\n" + "="*60)
            self.render_method("训练后区模型")
            self.render_method("="*60)
            self.bb_model, self.bb_encoder = self.train_model(X, y_bb, "bb")
            
            # 获取最后LOOKBACK_PERIODS期数据用于预测
            last_features = [X[-1]]
            
            # 进行预测
            rb_predicted = self.predict(last_features, "rb")
            bb_predicted = self.predict(last_features, "bb")
            
            # 输出结果
            self.render_method("\n" + "="*60)
            self.render_method("最终预测结果")
            self.render_method("="*60)
            self.render_method(f"前区号码: {sorted(rb_predicted)}")
            self.render_method(f"后区号码: {sorted(bb_predicted)}")
            
            # 输出最近一期开奖号码
            if not df.empty:
                last_row = df.iloc[-1]
                self.render_method("\n最近一期开奖号码:")
                if 'rb_1' in df.columns:
                    self.render_method(f"前区: {last_row['rb_1']}, {last_row['rb_2']}, {last_row['rb_3']}, {last_row['rb_4']}, {last_row['rb_5']}")
                if 'bb_1' in df.columns:
                    self.render_method(f"后区: {last_row['bb_1']}, {last_row['bb_2']}")
            
            total_time = time.time() - total_start
            self.render_method(f"\n总耗时: {total_time:.2f} 秒")
            
            # 添加结束分隔符
            self.render_method("\n" + "="*60)
            self.render_method("预测完成")
            self.render_method("="*60)
            
            return sorted(rb_predicted), sorted(bb_predicted)
            
        except Exception as e:
            self.render_method(f"\n{'*'*60}")
            self.render_method(f"发生错误: {str(e)}")
            self.render_method(f"{'*'*60}")
            self.render_method(traceback.format_exc())
            return None, None
    
    def get_recent_results(self, n: int = 5) -> pd.DataFrame:
        """获取最近n期开奖结果"""
        if self.df is None or self.df.empty:
            self.render_method("警告: 尚未加载数据或数据为空，无法获取最近开奖结果")
            return pd.DataFrame()
        
        # 确保n不超过数据量
        n = min(n, len(self.df))
        
        # 获取最近n期数据（原始数据中最新一期在最后）
        recent_df = self.df.tail(n).copy()
        
        # 选择需要的列
        result_df = recent_df[['date', 'issue', 'rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5', 'bb_1', 'bb_2']].copy()
        
        # 使用向量化操作替代apply（更高效且避免警告）
        # 创建前区号码
        red_balls = result_df[['rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5']].astype(str)
        result_df.loc[:, '前区号码'] = red_balls.agg(','.join, axis=1)
        
        # 创建后区号码
        blue_balls = result_df[['bb_1', 'bb_2']].astype(str)
        result_df.loc[:, '后区号码'] = blue_balls.agg(','.join, axis=1)
        
        # 重命名列（使用loc避免链式赋值警告）
        result_df = result_df.rename(columns={
            'date': '日期',
            'issue': '期号'
        })
        
        # 只返回需要的列
        return result_df[['日期', '期号', '前区号码', '后区号码']]