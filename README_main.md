## Plate & Leftover Segmentation 

MMSegmentation의 Mask2Former(Swin-S 백본)을 기반으로 **식판 위 접시/잔반/배경 3클래스**를 분류하는 파이프라인입니다.  
MMSegmentation 기본 환경 구축은 원본 `README_mmseg.md`(OpenMMLab 공식 가이드)와 동일하며, 본 문서는 그 위에 덧붙는 Food 프로젝트 절차만 기술합니다.

---

### 1. 환경 구축

1. NVIDIA CUDA 11.x + 최신 드라이버 (VRAM 24GB 이상 권장)
2. Anaconda 설치 후 아래 순서 실행
   ```bash
   conda create -n openmmlab python=3.8 -y
   conda activate openmmlab
   pip install -r requirements/runtime.txt     # MMSeg runtime
   pip install -r requirements/tests.txt       # 필요 시
   ```
3. 이후 `pip install -v -e .`로 MMSegmentation 개발 모드 설치 (원본 README와 동일)
4. OpenCL/OMP 에러 방지를 위해 `export OMP_NUM_THREADS=1` 등 환경변수 적용(기존 README 참조)

---

### 2. 데이터셋

데이터셋 구조:
```
data/food_dish_plate_clean/
 ├─ images/train/val/xxx.jpg
 └─ annotations/train/val/xxx.png
```

---

### 3. 학습

사용 config: `configs/mask2former/mask2former_swin-s_food-dish.py`
```
num_classes = 3
data_root = 'data/food_dish_plate_clean'
class_weight = [1.0, 1.0, 1.0, 0.1]
train_cfg = IterBasedTrainLoop(max_iters=40000, val_interval=1000)
```

실행 명령(GPU, 숫자 1을 GPU 수 만큼 교체):
```bash
bash tools/dist_train.sh \
  configs/mask2former/mask2former_swin-s_food-dish.py 1 \
  --work-dir work_dirs/mask2former_food_dish_plate_clean
```

> 최고 성능 모델(예: 12k iter, mIoU 86.5)과 학습 데이터 백업본(`data.tar.gz`)은 Google Drive 공유 폴더에 업로드되어 있습니다: <https://drive.google.com/drive/folders/1MYlWyUWyWuO8rLeXl4NJwXcQ2vL9EZws>. 저장소에는 포함하지 않습니다.

---

### 4. 검증 및 시각화

```bash
# 정량 평가 + 시각화
python tools/test.py \
  configs/mask2former/mask2former_swin-s_food-dish.py \
  best_mIoU_iter_12000.pth \
  --show-dir work_dirs/mask2former_food_dish_plate_clean/val_vis

# 샘플 비교(좌 원본 / 우 추론)
python tools/compare_inference.py \
  --config configs/mask2former/mask2former_swin-s_food-dish.py \
  --checkpoint best_mIoU_iter_12000.pth \
  --src-dir samples \
  --out-dir /home/work/layout/mmsegmentation/samples_compare \
  --device cuda:0 --opacity 0.6
```

추가로 단일 이미지 데모를 실행하려면 아래 예시를 그대로 활용하면 됩니다.

```bash
python demo/image_demo.py \
  data/unimib2016/original/20151127_114556.jpg \
  configs/mask2former/mask2former_swin-s_food-dish.py \
  best_mIoU_iter_12000.pth \
  --device cuda:0 \
  --out-file demo_outputs/unimib_20151127_114556.jpg \
  --opacity 0.6
```
(`demo_outputs/` 폴더는 미리 생성 필요)

---

### 5. 저장소 구성

```
├── configs/mask2former/..._food-dish.py
├── data/food_dish_final_bg_vis/          # 수동 검수된 오버레이
├── data/food_dish_plate_clean/           # 최종 학습/검증용 데이터
├── tools/
│   ├── extract_cleaned_food_ids.py
│   ├── copy_food_subset.py
│   ├── merge_datasets.py / split_dataset.py
│   └── compare_inference.py
└── work_dirs/mask2former_food_dish_plate_clean/
```

※ `data/`, `work_dirs/`는 대용량이므로 `.gitignore` 대상이며, 체크포인트는 별도 스토리지(Drive)로 공유.

---

### 6. 참고 / TODO

- `best_mIoU_iter_12000.pth` 및 baseline checkpoint → Drive 업로드 후 링크 업데이트 예정
- ONNX/TensorRT 내보내기 스크립트(필요 시) 추가
- 추가 실데이터(기타 트레이 이미지) 확장 검증 계획

---

### 7. 모델 / 데이터 다운로드 및 배치

1. Google Drive 링크(<https://drive.google.com/drive/folders/1MYlWyUWyWuO8rLeXl4NJwXcQ2vL9EZws>)에서  
   `best_mIoU_iter_12000.pth`와 `data.tar.gz`를 내려받습니다.
2. `data.tar.gz`를 압축 해제한 뒤 **폴더 전체를 레포지토리 루트의 `data/`에 위치시킵니다.**
   (기존 `data/`가 있다면 교체)
3. `best_mIoU_iter_12000.pth` 파일은 **레포지토리 최상위 경로(README가 위치한 곳)**에 둡니다.
4. 학습 재실행으로 생성된 최신 checkpoint를 별도로 저장하고 싶다면 `work_dirs/...` 아래를 사용해도 되지만,  
   문서의 검증/추론 명령어는 루트의 `best_mIoU_iter_12000.pth`를 참조하므로 반드시 동일한 파일명을 유지하세요.
