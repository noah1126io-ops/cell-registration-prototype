# Cell Registration Prototype

連続組織切片画像 2 枚と、それぞれの細胞セグメンテーションマスクを読み込み、後続の細胞対応推定と画像位置合わせの検討を行うための研究用ローカル Web アプリです。

このアプリは研究用プロトタイプであり、診断用途・治療方針決定・臨床判断には使用しないでください。

## 起動方法

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 現在できること

- fixed image のアップロードと表示
- moving image のアップロードと表示
- fixed mask のアップロードと表示
- moving mask のアップロードと表示
- mask を整数ラベル画像として読み込む

## 未実装の機能

- Cellpose によるセグメンテーション実行
- 細胞特徴量抽出の本実装
- 密度画像生成の本実装
- 細胞対応推定
- 画像位置合わせ
- 結果の可視化・エクスポート

