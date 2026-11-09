# 데이터셋: 수집, 형식, 단위

추가 가구 데이터를 수집하고 하나의 기준으로 가공한 기록입니다. 코드는 [`experiments/datasets.py`](../experiments/datasets.py)에 있습니다.

```bash
.venv/bin/python -m experiments.datasets fetch     # 원본 약 170 MB를 data/raw/에 받고 SHA-256 manifest 기록
.venv/bin/python -m experiments.datasets build     # data/processed/<collection>/<site>/ 생성
.venv/bin/python -m experiments.datasets summary
```

## 출처

| collection | 가구 수 | 부하 | PV | 라이선스·비고 |
|---|---|---|---|---|
| `tmy3` | 39 | NREL "Commercial and Residential Hourly Load Profiles for all TMY3 Locations" BASE (OpenEI dataset 961). 논문 참고문헌 [39]이며 기존 데이터의 원본 | PVGIS v5.3 `seriescalc`, 2019년, 최적 경사각, 손실 14% | 미국 기후대 1–8. 부하는 typical year, PV는 2019년 기상이라 같은 해가 아님 |
| `ausgrid` | 299 | Ausgrid Solar Home Electricity Data 2012-07 ~ 2013-06, 일반 소비(GC) | 같은 가구의 실측 발전량(GG) | CC BY 3.0 AU. Ausgrid 원본 페이지가 사라져 데이터 분석자(P. Haessig)의 사본을 사용. 결측이 있는 1가구 제외 |
| `korea` | 1 | 기존 `table1_load.npy` (KEPCO Seoul) | PVGIS v5.3, 2019년 | 원본이 이미 0.1 kWh로 반올림된 상태 |

관측소 좌표는 NOAA ISD `isd-history.csv`에서 가져옵니다(TMY3 파일명의 USAF 번호로 매칭).

## 형식

가구마다 폴더 하나:

```
data/processed/<collection>/<site>/
  load.npy         float32 (365, 24)  kWh/시간
  generation.npy   float32 (365, 24)  kWh/시간
  meta.json        출처 URL, 원본 SHA-256, 배율과 그 이유, 좌표, 시간대, 연간 통계
data/processed/index.json            전체 가구 요약
```

- **시간축**: 현지 표준시(일광절약시간 없음), day 0 = 1월 1일, hour h는 [h:00, h+1:00) 구간입니다. 평년이므로 2월 29일은 없습니다.
- **PVGIS 시간대 변환**: PVGIS는 UTC로 주므로 주별 표준시 오프셋만큼 이동합니다(El Paso는 산악 시간대). 발전 중심 시각이 각 지역의 태양 남중 시각과 맞는지 확인했습니다. 예: Phoenix 12.3시(남중 12.5시), Seoul 12.6시(남중 12.5시).
- **Ausgrid**: 회계연도(7월~6월)를 달력 순서(1~6월 2013, 7~12월 2012)로 재배열합니다. 남반구라 1월이 여름입니다.

## 단위와 정밀도: 판단과 근거

1. **저장은 물리 단위(kWh), 반올림 없음(float32)**
   기존 데이터를 역추적한 결과, `floor(원본 kWh × 5)`, 즉 0.5배 후 0.1 kWh 단위로 **내림**한 값이었습니다. 3개 지역 모두 8760시간이 100% 일치합니다.
   - 내림 때문에 연간 부하가 약 11% 사라집니다(Oakland: 0.5배 기준 3776 kWh → 저장값 3344 kWh).
   - PV는 0~4의 다섯 단계뿐입니다.

   스케줄러 입력에 정수가 필요하지 않으므로 반올림하지 않습니다. float32의 정밀도(상대 1e-7)는 0.001 kWh보다 훨씬 세밀합니다.
2. **스케줄러 내부 단위(0.1 kWh)로의 변환은 한 곳에서만** 합니다(`load_profile`, ×10). 행동 집합은 논문 그대로 0~2 kWh, 0.1 kWh 간격입니다. 수요가 소수여도 실행 가능 행동 계산(올림/내림)과 학습 mask가 실수를 그대로 처리합니다. 이 동작은 기존 테스트가 이미 다룹니다.
3. **기존 데이터(`data/load`, `data/generation`)는 그대로 둡니다.** `--preset paper`와 논문 재현이 이 데이터에 의존하기 때문입니다.
4. **배율**: 기존처럼 가구를 0.5배로 줄입니다(배터리 4 kWh, 구매 상한 2 kWh에 맞춘 "절반 규모 가구").
   - 0.5배로도 구매 상한 C_max = 2 kWh를 넘는 시간이 있으면 배율을 `1.95 / 원본 최대값`으로 더 낮춥니다. 이런 시간에는 배터리가 비면 수요를 충족할 수 없기 때문입니다(논문 가정 L_max ≤ C_max).
   - TMY3는 39곳 중 15곳이 해당합니다. 냉방 부하가 큰 남부 지역이며, Dallas는 0.5배 기준 367시간이 상한을 넘었습니다.
   - Ausgrid는 순간 피크가 커서 대부분 해당하고, 배율 중앙값은 0.37입니다. 가구별 배율은 `meta.json`에 있습니다.
5. **PV 0.5 kWp**: 논문과 같습니다. Ausgrid는 가구별 실제 설비 용량(1~3.4 kWp)을 0.5 kWp로 환산합니다.
6. **Ausgrid controlled load(CL) 제외**: 심야 전용 계량기·요금의 온수기 회로로, TOU/DC 스케줄링 대상이 아닙니다. 30분에 1.25 kWh씩 몰아 써서, 포함하면 가구 규모를 크게 줄여야 합니다.

## 사용

```bash
# TDAS: 한 가구
.venv/bin/python -m experiments.run_original --scenario 3 --sites "tmy3/*_TX_Dallas*"
# Co-TDAS: 가구마다 EMS 하나 (glob 가능)
.venv/bin/python -m experiments.run_original --algorithm Co-TDAS --sites "tmy3/*_TX_*,ausgrid/AU_NSW_customer00*"
# 학습 창: 7월부터 90일, 검증은 그다음 30일
.venv/bin/python -m experiments.run_original --scenario 3 --sites korea/KOR_Seoul --train-start 181 --train-days 90
```

Co-TDAS는 EMS가 요금 목록(8개)보다 많으면 TOU·DC 패턴을 반복합니다.

## 한계

- TMY3 부하는 시뮬레이션(EnergyPlus) 값이라 실측보다 매끄럽습니다. 실측 변동성은 Ausgrid가 대표합니다.
- Ausgrid 표본은 Ausgrid가 밝힌 대로 대표성이 없습니다(필터링된 300가구).
- 요금 계절(여름 4~9월)은 북반구 기준입니다. Ausgrid에도 같은 달력을 적용합니다.
- MPC 기준선(`MPC.py`)은 아직 기존 데이터만 읽습니다.
