# RO-GAN-using-Lightweight-GAN
Lightweight GAN(FastGAN)を用いてラグナロクオンラインのキャラクター画像を生成するGANです。<br>
学習済みモデルはONNX形式に変換して公開しています。<br><br>
<img src="https://user-images.githubusercontent.com/37477845/118026295-9e2d3e00-b39b-11eb-830f-9bd82ad48369.gif" width="400px">

# Requirement 
* lightweight-gan 0.20.0 or later
* torch 1.8.1 or later
* onnxruntime 1.7.0 or later

# Generation Example
<details>
<summary>生成例</summary>

<img src="https://user-images.githubusercontent.com/37477845/118026328-a84f3c80-b39b-11eb-85f3-27d7bc0f1a71.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026331-a8e7d300-b39b-11eb-9e0c-25449bdde1a5.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026336-a9806980-b39b-11eb-8944-f54398f0ff15.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026338-a9806980-b39b-11eb-92f5-30c1cfa5e688.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026339-aa190000-b39b-11eb-8695-599816158f2b.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026342-aab19680-b39b-11eb-9994-3a694d3dc627.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026351-ab4a2d00-b39b-11eb-8ce6-dc05f0facfef.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026353-ab4a2d00-b39b-11eb-8fe8-6016dcca8d67.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026359-ac7b5a00-b39b-11eb-8a3c-76b105f0d9b0.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026360-ad13f080-b39b-11eb-9515-1450ec8f86bc.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026364-adac8700-b39b-11eb-818b-7c72a1444d0a.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026365-ae451d80-b39b-11eb-86d2-ddaabdbc8973.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026368-ae451d80-b39b-11eb-83e6-dcb4d4ec5ee9.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026371-aeddb400-b39b-11eb-82e6-ac7e68827188.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026372-aeddb400-b39b-11eb-8d81-7daac9d00a5e.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026373-af764a80-b39b-11eb-9a41-64bd7bbda26e.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026377-b00ee100-b39b-11eb-9c97-e1acc8c4e0fb.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026379-b00ee100-b39b-11eb-8318-288eb46e9c2c.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026380-b0a77780-b39b-11eb-9fec-e4dd268ec037.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026384-b1400e00-b39b-11eb-9b1e-3cce0cc60ab7.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026387-b1400e00-b39b-11eb-891d-ad1c38f9d52b.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026388-b1d8a480-b39b-11eb-97ce-9ec4615709fb.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026391-b1d8a480-b39b-11eb-9049-0146948f2f84.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026392-b2713b00-b39b-11eb-96b7-11a119ddeafb.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026398-b309d180-b39b-11eb-85c8-5876e8ecee18.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026401-b3a26800-b39b-11eb-86f1-685248bd95de.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026980-67a3f300-b39c-11eb-8846-5d538c3210d4.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026983-683c8980-b39c-11eb-951d-5080b026d8e4.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026985-68d52000-b39c-11eb-86b5-b22226dc1e22.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026987-68d52000-b39c-11eb-8117-a6469b055866.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026992-696db680-b39c-11eb-9cdf-e725ceb52c33.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026994-6a064d00-b39c-11eb-94eb-e60f38b07204.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026995-6a064d00-b39c-11eb-9b0d-2b7495fe8483.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118026997-6a9ee380-b39c-11eb-9b77-2709283363ad.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027006-6bd01080-b39c-11eb-8eda-2daf64f5974e.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027009-6bd01080-b39c-11eb-9079-825ddea1db4d.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027014-6d013d80-b39c-11eb-8ba0-9382ea1b6d4f.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027015-6d013d80-b39c-11eb-946a-7b2992bc748d.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027300-bce00480-b39c-11eb-846c-8fad0a62a9a9.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027308-be113180-b39c-11eb-9a06-8a247be1b4e6.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027310-bea9c800-b39c-11eb-8a21-eccf17097c9b.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027311-bf425e80-b39c-11eb-9266-dc6365e8d333.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027314-bf425e80-b39c-11eb-90da-735ae179ac3e.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027317-bfdaf500-b39c-11eb-963e-3fb1a67c3f82.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027325-c1a4b880-b39c-11eb-9e6f-0fa947820447.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027328-c23d4f00-b39c-11eb-80f6-6433c491cebe.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027336-c36e7c00-b39c-11eb-9c75-4d7f7c8a443c.jpg" width="200px">
<img src="https://user-images.githubusercontent.com/37477845/118027338-c36e7c00-b39c-11eb-81eb-f077ffc75116.jpg" width="200px">

</details>

# Dataset
データセットは非公開です。

# Usage
以下のノートブックをColaboratoryで順に実施してください。<br>
ただし、データセットは本リポジトリに含めていないため、学習を実施する際にはご自身で収集・格納いただく必要があります。<br>
推論だけを試すのであれば「03_Inference_ONNX.ipynb」のみ実行してください。
* 01_Train_Lightweight_GAN.ipynb
* 02_Convert2ONNX.ipynb
* 03_Inference_ONNX.ipynb

# Reference
* [lucidrains/lightweight-gan](https://github.com/lucidrains/lightweight-gan)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
mediapipe-python-sample is under [Apache-2.0 License](LICENSE).
