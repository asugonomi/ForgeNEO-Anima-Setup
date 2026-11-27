# 実装メモ・実機検証の引き継ぎ

## 目的と境界

P-H3-Imageの文章生成を、ComfyUIのプロセス・ノードグラフに依存せず、Forge Neoの専用タブから実行する実験版です。通常txt2imgへの統合ではありません。UIだけを実装したダミーではなく、ロード・文章エンコード・サンプリング・VAE・保存の経路を実装しています。ただし、実モデルを使った通し実行は未実施です。

変更はこの拡張フォルダー内に限定してください。使用者のForgeは改変済みの可能性があるため、git reset、無断のgit pull、Torch / Gradio / transformersの一括更新、styles.csvの置換を行わないでください。

## ファイル構成

| ファイル | 責務 |
|---|---|
| `scripts/ph3_image.py` | WebUIの専用タブ登録だけを行うエントリーポイント |
| `ph3_native/ui.py` | Gradio UI、設定保存、診断表示。大きな重みはブラウザにアップロードしない |
| `contracts.py` | 固定シグマ列、入力検証、Seed列、ストリーム形状 |
| `headers.py` | 全重みをロードしないsafetensorsヘッダー検査。完全性・由来の証明ではない |
| `provenance.py` | 固定コミットのnativeコードとチェックサム検証、名前空間を分けたimport |
| `models.py` | P-H3用TextConnector、Qwen4Bの必要な5層、単一フレームVAEデコーダー |
| `loading.py` | Forge量子化形式への接続、厳格な重みキー照合、FP32保持、mmap復帰、ModelPatcher生成 |
| `sampling.py` | 固定8ステップEuler。内部の画像・音声潜在を両方更新 |
| `engine.py` | Forge共有ロック、段階別ロード、CPUキャッシュ、キャンセル、OOM再試行、後始末 |
| `output.py` | PNG parametersとJSONの原子的ファイル置換による保存 |
| `diagnostics.py` | 対象API・CUDA・VRAMモード・モデルヘッダーの診断。生成検証とは区別 |
| `tools/setup_native.py` | nativeソースと公式tokenizerだけを取得。pip等は呼ばない |
| `tools/download_models.py` | 明示操作による公式3重み取得。全リポジトリは取得しない |

## 重要な実装上の判断

TextConnectorの射影にはbiasがなく、学習済みsinkベクトルは先頭トークンを**置き換え**ます。トークンを追加しません。Qwenの取得箇所はhidden_states[13, 19, 25, 31, 36]です。最後の36は最終正規化後、それ以外は該当層の入力側です。

VAEは静止画用で、patch_size_t=1、画像フレームの座標は `2 * (0.5 / 7) - 1` です。動画用VAEの1フレーム出力を単に切り取って代用していません。量子化から除外されたFP32パラメータはロード後に復元します。

スカラー等を含めた精度、演算順序、乱数消費順、量子化カーネルは参照実装との実機数値比較が必要です。同一SeedでComfyUIと画素が一致するとは定義していません。

`load_models_gpu` の `memory_required` と `minimum_memory_required` を指定しますが、これを厳密なVRAM上限と扱わないでください。`force_full_load=False` を維持し、highvram / gpu-onlyモードは診断で拒否します。モデルを一括 `.to("cuda")` する修正は禁止です。

`using_forge_operations` の一時的なクラス置換も、Forge共有生成ロックを保持した状態で行います。上流H3拡張の `_hook_detection`、`_hook_components` 等の全体向けパッチは呼びません。

停止要求はEventを通じて伝え、共有ロックを待つ前にそのEventを初期化します。実行中に停止要求を受けてから再び初期化してはいけません。OOMの例外処理を抜けた後にGC・GPU解放を行い、例外tracebackがGPUテンソルを保持し続けるのを避けます。

## 最優先の実機確認

1. 改変済みForgeをコピーして検証環境を作成し、現行commit、起動引数、Python、Torch、CUDAビルド、comfy-kitchen、transformers、Gradioを記録します。依存関係を先に更新して問題を隠さないでください。
2. 公式3重みの出所・リビジョンを確認します。対応するホストの量子化キー変換が `_strict_load` に不足・余剰を返さないか確認します。エラーを無視するstrict=False継続には変更しないでください。
3. 固定Seed・短い英語文章・2048×2048・1枚で、文章処理→DiT→VAE→PNG保存まで通します。画像が正常であることも確認します。
4. 初回と2回目を分けて時間・VRAM・RAMを記録します。実測はTorch allocated / reservedに加え、他プロセス・ドライバーも含む値を区別して記載します。
5. 同じSeed・条件で再現性、連続2枚、別プロンプト、CPUキャッシュOFF、手動解放を確認します。
6. 文章処理中、DiT中、VAE中の停止と、通常txt2imgに戻った後の生成を確認します。
7. 2912×2912に上げてVRAM16GBで実測します。OOM時の自動再試行がSeed・解像度を維持すること、失敗が成功表示に変わらないことを確認します。
8. 参照実装と比較し、TextConnectorの精度、VAEタイル境界、スケジュールと速度の符号、非有限値を調べます。色や構図が出るだけでは正しい移植の証明にはなりません。

失敗があった場合、改修はこの拡張内で完結するアダプターを優先してください。外部依存の変更が避けられない場合は、何をなぜ変えるかを提示し、使用者の承認とバックアップを得てから行います。

## 将来の追加機能

画像編集は、UIに画像入力を足すだけでは完成しません。Qwenのvision tower、DeepStack、画像トークンの位置・modalityタグ、VAEエンコード、参照潜在の配置、適切なメモリ見積もりが必要です。Prompt Upsamplerにも別のモデルと実行段階が必要です。初版の未対応機能を対応済みとして表示しないでください。
