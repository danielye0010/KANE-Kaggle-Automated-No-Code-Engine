import os
import pandas as pd
import torch
from autogluon.multimodal import MultiModalPredictor

def main():
    # 设置全局线程数为 1（避免多进程问题）
    torch.set_num_threads(1)

    # 读取指定路径的训练和测试数据
    train_data = pd.read_csv(r"D:\Desktop\python project\pythonProject\data\train.csv")
    test_data = pd.read_csv(r"D:\Desktop\python project\pythonProject\data\test.csv")

    # 定义目标标签
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # 创建文件夹用于保存模型
    models_dir = r"D:\Desktop\python project\pythonProject\models"
    os.makedirs(models_dir, exist_ok=True)

    # 训练并保存多个模型
    for label in label_columns:
        try:
            print(f"正在训练模型以预测标签: {label}...")
            # 为每个标签单独创建一个 MultiModalPredictor
            predictor = MultiModalPredictor(label=label, problem_type='binary', eval_metric='roc_auc')

            # 训练模型
            predictor.fit(train_data=train_data, time_limit=3600)

            # 保存训练好的模型
            model_path = os.path.join(models_dir, f"predictor_{label}")
            predictor.save(model_path)
            print(f"模型 {label} 已成功保存至：{model_path}")

        except Exception as e:
            print(f"训练模型 {label} 失败，错误信息: {str(e)}")

    print("所有模型已训练完毕并保存。")

    # 加载模型并生成预测结果
    predictions = pd.DataFrame()
    predictions['id'] = test_data['id']

    # 提取只包含特征列的测试数据，假设 'comment_text' 是唯一的输入特征
    test_features = test_data[['comment_text']]

    for label in label_columns:
        try:
            print(f"正在加载模型以预测标签: {label}...")

            # 加载已保存的模型
            model_path = os.path.join(models_dir, f"predictor_{label}")
            predictor = MultiModalPredictor.load(model_path)

            # 生成当前标签的预测概率
            predictions[label] = predictor.predict_proba(test_features)[1]  # [1] 表示获取预测为 1 的概率

        except Exception as e:
            print(f"加载或预测模型 {label} 失败，错误信息: {str(e)}")

    # 准备提交文件
    submission_file = os.path.join(r"D:\Desktop\python project\pythonProject\data", 'submission.csv')
    predictions.to_csv(submission_file, index=False)

    print(f"提交文件已生成：{submission_file}")

# 确保在 Windows 上多进程调用时不会报错
if __name__ == '__main__':
    main()
