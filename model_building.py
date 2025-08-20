import os
import pandas as pd
import numpy as np
from pycaret.classification import *
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
import logging

# 配置日志
logging.basicConfig(filename='model_building.log', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_models():
    """获取当前环境可用的模型列表"""
    all_models = models()
    available = []
    for model in ['lightgbm', 'rf', 'catboost', 'gbc']:
        if model in all_models.index:
            available.append(model)
    return available


def load_and_preprocess():
    """简化版数据加载"""
    print("正在加载数据...")
    df = pd.read_csv('outputs/data/loan_quick.csv')

    # 基础预处理
    df['is_default'] = (df['loan_status'] == 'Charged Off').astype(int)
    df = df[['loan_amnt', 'int_rate', 'grade', 'dti', 'fico_range_low', 'is_default']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['grade'] = df['grade'].astype(str).str.strip()

    # 采样控制数据量
    if len(df) > 100000:
        df = df.sample(100000, random_state=42)
    return df


def manual_calibration(model, X_train, y_train, X_val, y_val):
    """手动实现概率校准"""
    try:
        # 获取模型在验证集上的预测概率
        val_probs = model.predict_proba(X_val)[:, 1]

        # 使用保序回归手动校准
        from sklearn.isotonic import IsotonicRegression

        # 训练保序回归模型
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(val_probs, y_val)

        # 创建校准后的预测函数
        def calibrated_predict_proba(X):
            raw_probs = model.predict_proba(X)[:, 1]
            calibrated_probs = iso_reg.transform(raw_probs)
            # 返回二维数组格式
            return np.vstack([1 - calibrated_probs, calibrated_probs]).T

        # 创建一个包装器类来保存校准后的模型
        class CalibratedModel:
            def __init__(self, base_model, calibrator):
                self.base_model = base_model
                self.calibrator = calibrator

            def predict_proba(self, X):
                return calibrated_predict_proba(X)

            def predict(self, X):
                probs = self.predict_proba(X)
                return (probs[:, 1] > 0.5).astype(int)

        calibrated_model = CalibratedModel(model, iso_reg)
        print("手动保序回归校准成功")
        return calibrated_model

    except Exception as e:
        print(f"手动校准失败: {e}")
        return model


def calibrate_probabilities(model, X_train, y_train):
    """对模型进行概率校准"""
    try:
        # 检查模型是否支持概率校准
        if not hasattr(model, 'predict_proba'):
            print("模型不支持概率预测，跳过校准")
            return model

        print("正在进行概率校准...")

        # 方法1: 尝试使用不同的校准方法
        try:
            # 使用sigmoid方法，设置较小的cv值
            calibrated_model = CalibratedClassifierCV(
                model,
                method='sigmoid',
                cv=3,  # 使用3折交叉验证
                ensemble=False
            )
            calibrated_model.fit(X_train, y_train)
            print("Sigmoid校准成功")
            return calibrated_model

        except Exception as e1:
            print(f"Sigmoid校准失败: {e1}")

            # 方法2: 尝试保序回归
            try:
                calibrated_model = CalibratedClassifierCV(
                    model,
                    method='isotonic',
                    cv=3,
                    ensemble=False
                )
                calibrated_model.fit(X_train, y_train)
                print("保序回归校准成功")
                return calibrated_model

            except Exception as e2:
                print(f"保序回归校准失败: {e2}")

                # 方法3: 手动校准（分割数据）
                from sklearn.model_selection import train_test_split
                X_train_cal, X_val_cal, y_train_cal, y_val_cal = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
                )

                # 重新训练基础模型（避免分类特征问题）
                base_model = clone(model)
                base_model.fit(X_train_cal, y_train_cal)

                # 手动校准
                return manual_calibration(base_model, X_train_cal, y_train_cal, X_val_cal, y_val_cal)

    except Exception as e:
        print(f"所有校准方法都失败: {e}")
        return model


def build_robust_model():
    """稳健建模流程"""
    try:
        # 1. 数据准备
        data = load_and_preprocess()
        print(f"数据准备完成，样本量: {len(data)}")

        # 2. 设置环境
        exp = setup(
            data=data,
            target='is_default',
            session_id=42,
            train_size=0.7,  # 留出更多数据用于校准
            normalize=True,
            fix_imbalance=True,
            fold=3,
            use_gpu=False,
            verbose=False,
            index=False
        )

        # 3. 获取可用模型
        available_models = get_available_models()
        print(f"可用模型: {available_models}")

        if not available_models:
            raise ValueError("没有可用的模型，请检查安装")

        # 4. 模型训练
        best = compare_models(
            include=available_models,
            n_select=1,
            sort='AUC',
            cross_validation=False
        )

        print(f"最佳模型: {type(best).__name__}")

        # 5. 概率校准
        X_train = get_config('X_train')
        y_train = get_config('y_train')

        calibrated_model = calibrate_probabilities(best, X_train, y_train)

        # 6. 保存模型
        save_model(calibrated_model, 'outputs/models/final_model_calibrated')
        print(f"\n校准后的模型保存成功: outputs/models/final_model_calibrated.pkl")

        save_model(best, 'outputs/models/final_model_original')
        print(f"原始模型保存成功: outputs/models/final_model_original.pkl")

        return calibrated_model, best

    except Exception as e:
        print(f"\n建模失败: {str(e)}")
        logger.error(f"建模失败: {str(e)}")
        return None, None


# 添加测试校准效果的函数
def test_calibration_effect(original_model, calibrated_model, X_test, y_test):
    """测试校准效果"""
    from sklearn.metrics import brier_score_loss, log_loss

    # 原始模型预测
    original_probs = original_model.predict_proba(X_test)[:, 1]
    original_brier = brier_score_loss(y_test, original_probs)
    original_log = log_loss(y_test, original_probs)

    # 校准后预测
    calibrated_probs = calibrated_model.predict_proba(X_test)[:, 1]
    calibrated_brier = brier_score_loss(y_test, calibrated_probs)
    calibrated_log = log_loss(y_test, calibrated_probs)

    print(f"\n=== 校准效果评估 ===")
    print(f"原始模型 Brier Score: {original_brier:.4f}")
    print(f"校准后 Brier Score: {calibrated_brier:.4f}")
    print(f"原始模型 Log Loss: {original_log:.4f}")
    print(f"校准后 Log Loss: {calibrated_log:.4f}")


if __name__ == "__main__":
    os.makedirs('outputs/models', exist_ok=True)
    print("=== 贷款违约预测建模 ===")

    calibrated_model, original_model = build_robust_model()

    if calibrated_model:
        print("\n=== 建模成功 ===")
        print(f"校准后模型: {type(calibrated_model).__name__}")
        print(f"原始模型: {type(original_model).__name__}")

        # 测试校准效果
        try:
            X_test = get_config('X_test')
            y_test = get_config('y_test')
            test_calibration_effect(original_model, calibrated_model, X_test, y_test)
        except:
            print("无法获取测试数据，跳过校准效果评估")
    else:
        print("\n=== 建模失败 ===")
        print("请检查日志文件 model_building.log")