## Fine-tuning Notes

### 진행 사항

- 최근 fine-tuning checkpoint
  - `work_dirs/mask2former_food_dish_mixed_ft/best_mIoU_iter_200.pth`

- 기존 baseline checkpoint(Fine-tuning 전):
  - `best_mIoU_iter_12000.pth`

### Fine-tuning 목적

실서비스 이미지에서 발생한 다음 오탐을 줄이기 위해 fine-tuning을 진행함.

- 그릇 반사 영역이 leftover로 잡히는 문제
- 소스 자국/얼룩이 leftover로 과하게 잡히는 문제
- 실제보다 leftover 영역이 넓게 잡히는 문제

### 데이터 구성

```text
finetuning/
└── correction_set/
    ├── images/
    ├── masks/
    └── overlays/
```

Correction set은 기존 모델 inference 결과 중 오탐 사례를 선별한 뒤, mask를 수작업 수정하여 생성함.

### Fine-tuning 실행

```bash
cd segmentation_plate_leftover-main

python tools/train.py \
  configs/mask2former/mask2former_swin-s_food-dish-empty-ft.py \
  --work-dir work_dirs/mask2former_food_dish_mixed_ft
```

학습 결과:

```text
work_dirs/mask2former_food_dish_mixed_ft/
```

## Github - toy_app AI 파트 수정 사항

```text
services/mmseg_service.py
```

다음 기준을 코드단에서 조정 가능.

- MIN_PLATE_AREA_RATIO
- MIN_PLATE_PIXELS
- MIN_LARGEST_PLATE_COMPONENT_RATIO
- MIN_PLATE_FILL_RATIO
- MIN_LEFTOVER_NEAR_PLATE_RATIO

식판 인식 실패 시 `-1.0`을 반환하며, Backend에서는 재촬영(RETAKE)으로 처리함.