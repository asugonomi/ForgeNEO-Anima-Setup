# ForgeNEO Anima セットアップ

Windows x64 向けの日本語 Forge Neo 環境です。最初に [Install-ForgeNEO.bat](https://github.com/asugonomi/ForgeNEO-Anima-Setup/releases/download/v1.0.0/Install-ForgeNEO.bat) を保存し、十分な空き容量のあるフォルダーでダブルクリックしてください。

BATと同じ場所に ForgeNEO-Anima フォルダーができます。セットアップ後は ForgeNEO.exe で起動し、ForgeNEO-Update.exe で更新します。コンソールの文字を選択してコピーでき、ログは logs に保存されます。

公式 Anima Base v1.0、Qwenテキストエンコーダー、VAEは [CircleStone Labs](https://huggingface.co/circlestone-labs/Anima) から取得します。FnMoChIrMrgBl5_int8ConvRot は配布元登録前は保留になり、Anima Baseで起動します。登録後は更新メニューのモデル同期で追加できます。

## FnMoの配布元登録

リポジトリの models.json の url に、公開Hugging Faceファイルの https://huggingface.co/所有者/モデル名/resolve/コミット/ファイル名 形式のURLを指定します。ダウンロード時に使用中の原本のサイズとSHA-256を照合します。同じモデルを手元に持っている場合は、インストール先の models.json にも同じ形式で指定できます。

## 構成

使用中のForge本体、外部拡張30件と有効・無効設定、ワイルドカード、日本語翻訳、タグ辞書、styles.csv、プリセット、ADetailer・SAM・アップスケーラーを復元します。内容は inventory.json で確認できます。追加のLoRAやワイルドカードが参照する個別LoRAは、必要に応じて forge/models/Lora に置いてください。

Windows x64、インターネット接続、NTFSのローカルドライブを想定しています。初回は空き容量35 GiB以上を推奨します。GPU用PyTorchの導入後に演算を試し、利用できるバックエンドを選びます。CPUでは生成に時間とメモリーが必要です。機種やドライバーによって、個別拡張やINT8演算の対応状況が異なります。

コードとモデルの利用条件はそれぞれの配布元に従ってください。
