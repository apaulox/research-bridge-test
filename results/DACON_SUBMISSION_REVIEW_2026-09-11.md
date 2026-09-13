**DACON 제출 ZIP 중간 점검 — 2026-09-11**

**후속 규정 확인에 따른 정정:** 현재 [DACON 공식 규칙](https://dacon.io/competitions/official/236749/overview/rules)은 파일별 독립 예측을 요구하며, 다른 파일의 정보·예측값·통계를 이용한 예측·보정을 금지한다. 따라서 서로 다른 파일 사이의 attention을 유지한 **`submit_clam_official_v2.zip`과 `submit_eat75_clam25_fmc_v3.zip`은 현재 명시된 원칙에 맞지 않는다.** 아래 점수는 과거 제출 결과의 기록일 뿐, 해당 구성을 최종 제출 후보로 권장하는 의미가 아니다. seed 고정은 파일 간 의존성을 없애지 않는다. 같은 파일 내부 구간을 나누고 결합하는 방식은 허용된다. 최초 점검에서 이 규정과 구현을 연결하지 못한 부분을 정정한다. 과거 채점 여부나 운영진의 개별 제재 판단을 새로 확인한 것은 아니다.

가장 중요한 구분은 다음과 같다. **직접 신경망 가중치를 학습한 것은 MusicDET SpecNF다. EAT·CLAM 등은 공개 가중치를 사용했고, MusicDET 및 CLAM의 확률 보정기는 따로 학습했다.** 공식 가중치 사용, 직접 신경망 학습, 직접 보정기 학습을 서로 다른 항목으로 기록한다.

범위는 현재 Windows·WSL에서 찾은 ZIP과 두 이전 작업의 제출 결과, 사용자가 남긴 초기 1–12번 실험 기록이다. **점수 기록이 있는 17개 실험**, EAT+MusicDET의 실행 실패·수정판, EAT+CLAM의 실패판, 준비용 ZIP을 포함한다. DACON 계정의 제출 목록을 직접 내보낸 자료는 아니므로, 기록에 없는 제출까지 전부 확인했다고 주장하지 않는다. `music_eat`, `voice_antideepfake`처럼 확장자 없는 이름은 과거 기록명을 유지했다.

초기 baseline 외 2–11번 ZIP 실물은 현재 조회한 폴더에서 찾지 못했다. 해당 실험의 설정·점수는 과거 사용자 기록에 근거하며, 정확한 가중치 해시·분기 연결을 다시 검증한 것으로 취급하지 않는다. 이후 ZIP은 내부 `script.py`, 모델 출처 문서, 보정 JSON, 학습 로그를 대조했다. 이번에는 모델 재학습이나 성능 평가를 실행하지 않았다. dummy/sample 폴더의 예측은 성능 근거에서 제외했다.

근거 보관: [초기 실험 메모](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/earlier_experiment_notes.txt), [제출 결과 대화 기록](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/conversation_evidence.json), [Windows ZIP 목록·내용](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/windows/archive_inventory.json), [WSL ZIP 목록·내용](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/wsl/archive_inventory.json), [ZIP 간 변경 비교](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/archive_comparisons.json).

**1. 공통 baseline과 데이터의 정답 정의**

| 역할 | 모델·입력 | 학습 데이터와 성격 | 이번 프로젝트의 학습 여부 |
|---|---|---|---|
| Voice/Music 존재 | PANNs Cnn14, 원본 mix를 구간별 처리 | AudioSet: 발화·가창·악기·환경음 등 527개 소리 사건. real/fake 판별 데이터가 아니라 소리 종류 태깅 데이터 | DACON 제공 공개 checkpoint 사용. 추가 학습 없음 |
| 음원 분리 | HTDemucs `955717e8-8726e21a` | MUSDB-HQ + 추가 800곡. mix와 vocals/drums/bass/other stem의 대응을 배우는 음악 분리 데이터 | 공식 `htdemucs` 가중치. 추가 학습 없음 |
| 성분 진위 | DF-Arena-1B, 분리 vocals와 나머지 반주 합산 stem 각각 입력 | ASVspoof 2019·2024, Codecfake, LibriSeVoc, DFADD, CTRSVDD, SpoofCeleb, MLAAD, EnvSDD. 발화 위조 중심에 가창·환경음 위조를 포함하는 혼합 학습 | `Speech-Arena-2025/DF_Arena_1B_V_1` 공개 가중치. 추가 학습 없음 |

PANNs checkpoint는 `Cnn14_mAP=0.431.pth`다. DF-Arena는 revision `fb6ce85de12c2c5a509d89114adaf827dd75f49f`로 출처가 기록돼 있다. DF-Arena 공개 카드가 열거한 데이터셋은 확인했지만, 학습 실행별 정확한 파일 목록·비중은 미확인이다. [PANNs 공식 저장소](https://github.com/qiuqiangkong/audioset_tagging_cnn), [HTDemucs 공식 저장소](https://github.com/facebookresearch/demucs), [DF-Arena 모델 카드](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1).

baseline은 16kHz 오디오를 64,600 sample, 약 4.0375초 구간으로 처리한다. 짧으면 반복하여 채우고, 성분별 구간 fake 확률은 최댓값으로 집계한다. PANNs에는 32kHz로 변환한 원본 구간을 넣고, 관련 클래스·구간의 최댓값으로 존재 확률을 만든다. HTDemucs는 vocals를 Voice, 나머지 stem 합을 Music으로 전달한다. 분리 모델은 AI 진위 판별기가 아니다.

기본 FILE 결합은 `max(VP × VF, MP × MF)`다. 존재 확률이 낮은 성분의 진위 점수를 억제하고, 둘 중 강한 fake 신호를 채택하는 설계다. 논리적 OR에 대응하는 휴리스틱이며, 통계적으로 보정된 파일 확률이라는 보장은 없다. **공식이 같아도 VF/MF가 바뀌면 FILE 출력과 순위가 바뀐다.** [실제 baseline 코드](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/wsl/baseline_submit/script.py).

DACON에서 **보컬은 Voice, 반주·악기음은 Music**이다. 둘 중 하나라도 AI가 생성했으면 FILE은 fake다. 따라서 원본 노래 전체의 real/fake 라벨로 학습한 탐지기를 MUSIC_FAKE에 연결할 때, 보컬 합성 신호를 반주 합성으로 잘못 해석할 수 있다. 실제 반주 위 AI 보컬은 Voice fake / Music real이다. 현재 높은 CPS는 존재 탐지를 뒷받침하지만, 분리 품질이나 성분 진위의 독립성을 증명하지 않는다. [DACON 공식 문제 정의](https://dacon.io/competitions/official/236749/overview/description).

**2. 점수 기록이 있는 모든 실험**

아래 점수는 제출 결과 또는 사용자 메모의 수치를 그대로 보존했다. 초기 소수 넷째 자리 점수는 근삿값이다. 원본 ZIP 미확보 실험의 FILE 출력 고정 여부는 메모만으로 확정하지 않았다.

| 번호 | 제출명·기록명 | 점수 | 실제 변경·목적 | 학습·데이터 | 판단 |
|---|---|---:|---|---|---|
| 1 | `baseline_submit.zip` | 약 0.6909 | PANNs + HTDemucs + DF-Arena 양쪽 성분 + MAX | 위 공통 공개 모델들. 직접 학습 없음 | 전체 비교 기준 |
| 2 | `submit_music_const.zip` | 약 0.6739 | MUSIC_FAKE만 0.5로 두어 Music 성능 분해 | baseline 동일, 추가 데이터 없음 | 진단 제출. FILE 포함 다른 출력 고정이 역산 조건 |
| 3 | `submit_voice_const.zip` | 약 0.6413 | VOICE_FAKE만 0.5로 두어 Voice 성능 분해 | baseline 동일, 추가 데이터 없음 | 진단 제출. 다른 출력 고정이 역산 조건 |
| 4 | `file_from_voice_only` | 약 0.6883 | FILE을 `VP×VF`로 변경하여 음악 항의 순효과 확인 | baseline 동일, 추가 데이터 없음 | 제거 후 소폭 하락. 음악 항의 전체 순기여는 작았으나, 유용성이 없다는 뜻은 아님 |
| 5 | `submit_music_diag_v3.zip` | 약 0.6690 | 음악 판별을 SONICS SpecTTTra-β-5s로 교체. 반주 stem·반복 채움·max 집계 | SONICS 공개 모델 사용 기록. 직접 학습 기록 없음 | baseline보다 낮음. 모델과 학습 데이터 변경이 함께 포함됨 |
| 6 | `submit_music_mix_mean.zip` | 약 0.6667 | SONICS 유지. stem→원본 mix, 반복→zero-pad, max→mean | 같은 SONICS 가중치, 추가 학습 없음 | 세 변경의 묶음이 개선되지 않음. 각 변경의 효과 분리 불가 |
| 7 | `submit_voice_from_mix.zip` | 약 0.6828 | Voice DF-Arena 입력을 vocals stem→원본 mix로 변경 | baseline 공개 가중치, 추가 학습 없음 | 이 설정의 원본 mix 입력이 불리했음. 모든 입력 문제를 배제하지는 못함 |
| 8 | `submit_music_eq_voice.zip` | 약 0.6662 | MUSIC_FAKE 컬럼을 VOICE_FAKE로 대체. 메모상 FILE 원본 유지 | baseline 동일, 추가 데이터 없음 | Voice 점수를 Music 정답에 그대로 쓰는 방식 실패. 성분 라벨 동일 가설 지지 안 됨 |
| 9 | `voice_antideepfake` | 약 0.6976 | Voice를 AntiDeepfake XLS-R-1B로 교체, vocals stem 유지 | 연구팀 공개 74,650시간 후속학습 모델 사용 기록. 우리 추가 학습 없음 | 제출 구성은 개선. Voice/File 개선량의 정확한 분해는 재확인 필요 |
| 10 | `music_from_mix` | 약 0.6939 | Music DF-Arena 입력만 반주 stem→원본 mix | baseline 공개 가중치, 추가 학습 없음 | 입력 경로 변경에 소폭 이득. 다른 탐지기에도 mix가 최적이라는 증거는 아님 |
| 11 | `music_eat` | 약 **0.7224** | 음악 판별을 DeepFense EAT-large + AASIST로 교체 | FMC 학습 공개 모델 사용 기록. 우리 추가 학습 없음 | 점수 기록 중 최고. 초기 ZIP과 후속 EAT 런타임의 완전 동일성은 미확인 |
| 12 | `submit_musicdet_fmc_v1.zip` | **0.7108396402** | 음악 DF-Arena→직접 학습 SpecNF. 원본 mix 중앙 4.0375초, 로지스틱 보정. 새 MF가 FILE에도 반영 | MusicCaps real 4,120개로 가중치 학습. FMC dev로 모델 선택·보정 | baseline 개선. 구조·학습·입력·보정의 묶음 효과 |
| 13 | `musicdet_fmc_file_probe.zip` | **0.6317039259** | 12번의 FILE_FAKE만 0.5로 고정 | 모델·가중치·입력·보정 동일. 추가 학습 없음 | 진단 제출. 낮은 총점은 의도된 결과 |
| 14 | `submit_eat75_clam25_fmc_v2.zip` | **0.7031253545** | 공식 EAT75 + 공식 CLAM25 logit 결합, CLAM 배치1·FMC 반전 보정. v1의 timm 제거 | 공개 EAT·MoM-CLAM 가중치. CLAM 보정만 직접 학습 | baseline보다 높으나 기존 EAT 기록보다 낮음 |
| 15 | `submit_clam_official_v1.zip` | **0.6204539259** | 공식 CLAM만 Music에 사용. 배치1, 공식 방향 sigmoid, FMC 보정 없음 | 공개 MoM checkpoint. 추가 학습 없음 | 이 공식 가중치의 단일 파일 추론 구성은 성능 부족 |
| 16 | `submit_clam_official_v2.zip` | **0.6358182116** | 15번 대비 head 배치16·float32 sigmoid, seed42 그룹 고정 | 같은 공식 가중치. 추가 학습·보정 없음 | +0.0153642857. 공식 배치 동작 재현으로 개선됐지만 baseline 미달 |
| 17 | `submit_eat75_clam25_fmc_v3.zip` | **0.7177824974** | 14번 대비 CLAM head 배치16 + 새 배치 출력으로 FMC 보정 재학습 | 같은 EAT·CLAM 가중치. 보정기만 재학습 | +0.0146571429. 기존 EAT 기록에는 약 0.00462 부족 |

확인된 실행 시간: EAT 기록 17분 5초, MusicDET 15분 44초, MusicDET probe 15분 50초, EAT+CLAM v2 30분 37초, v3 30분 11초, CLAM 단독 v1 29분 59초, v2 29분 44초. 초기 baseline의 정확한 실행 시간은 이번 자료에서 확정하지 않았다.

**3. 실패판·수정판·별칭을 포함한 ZIP 이력**

| 파일 | 실제 내용·버전 관계 | 제출 상태 근거 |
|---|---|---|
| `submit_eat75_musicdet.zip` | WSL `submit_eat75_musicdet25_fmc_v1.zip`와 모든 entry의 CRC·크기 동일. EAT 런타임이 timm을 요구하지만 requirements에 누락 | 이후 `No module named 'timm'` 오류 보고가 이어짐. 원래 업로드 이름의 정확한 대응은 대화 순서에 근거 |
| WSL `submit_eat75_musicdet25_fmc_v2.zip` | v1에서 requirements에 `timm==1.0.22`만 추가. script·가중치 동일 | 사용자가 `eat75musicdet25.zip`의 libtorchaudio 오류 보고. 과거 Windows 동일 이름 파일은 이후 교체됨 |
| 현재 Windows `submit_eat75_musicdet25.zip` | WSL `submit_eat75_musicdet25_dependencyfix.zip`와 entry 전체 동일. timm 유틸리티를 로컬 구현으로 대체하고 설치 의존성 제거 | 현재 파일은 수정판. 수정판의 정상 채점 점수는 확보하지 못함 |
| WSL `submit_eat75_musicdet25_dependencyfix.zip` | 직전 행과 같은 내용. 새 학습 모델이 아니라 패키징 수정 | 생성·실행 검증 기록 있음. 정상 제출 점수 미확인 |
| `submit_eat75_clam25_fmc_v1.zip` | EAT+CLAM 배치1·음수 보정 원형. requirements에 timm 포함 | libtorchaudio 오류 보고 후 v2 제작. 가중치 학습 실패가 아니라 실행 환경 단계 오류 |
| `musicdet_fmc_starter.zip` | 약 64KB의 초기 코드 준비물. 완성 탐지기 checkpoint·제출 루트 script 없음 | 완성된 성능 제출로 세지 않음 |
| `open.zip` | DACON 배포 컨테이너. 내부 baseline_submit.zip의 원본 배포물 | 별도 모델 실험으로 세지 않음 |

동일성은 이번에 ZIP 중앙 디렉터리의 **entry별 CRC와 크기**, 작은 코드 파일의 내용으로 확인했다. 전 가중치의 SHA-256을 이번에 다시 계산한 것은 아니다. 과거 manifest에 저장된 SHA-256 검증은 별도 근거다. 파일 이름만으로 과거 제출과 현재 파일을 동일시하면 안 된다. 특히 `submit_eat75_musicdet25.zip`은 현재 수정판이다.

**4. 모델별 checkpoint·학습 데이터·선택 이유**

**SONICS SpecTTTra-β-5s — 공식 모델 사용 기록, 직접 학습 없음**

SONICS는 데이터셋 이름이며 SpecTTTra가 탐지 모델이다. 공개 배포 ID는 `awsaf49/sonics-spectttra-beta-5s`다. 실제 노래와 Suno/Udio 생성 노래 등 약 97,000곡 규모의 전체곡 진위 탐지 데이터로 개발된 모델이다. 장시간 노래의 패턴을 다루는 계열에서 5초용 경량 모델을 사용했다. [공식 checkpoint](https://huggingface.co/awsaf49/sonics-spectttra-beta-5s), [SONICS 논문](https://arxiv.org/abs/2408.14080).

음악 전용 탐지기를 음성 위조까지 함께 다루는 DF-Arena 대신 넣어 보려는 취지는 타당하다. 다만 전체곡 단위 real/fake와 DACON의 반주 성분 real/fake는 같은 정답이 아니다. 반주 stem을 넣으면 보컬 중심 학습과 달라질 수 있고, 원본 mix를 넣으면 보컬 진위가 Music 점수에 섞일 수 있다.

5번은 기존 분기 입력과 집계를 유지하려는 비교였고, 6번은 원본 음악 입력·자연스러운 패딩·평균 집계로 학습 조건에 더 가까워지기를 기대한 변경이다. 후자의 구체적인 공식 전처리 일치 여부는 초기 ZIP 미확보로 재검증하지 못했다. 세 요소를 같이 바꿨으므로 원인 분리는 불가능하다.

과거 메모의 **“97,164곡 전부 no_vocal=False이므로 반주를 한 번도 학습하지 않았다”는 단정은 보류**한다. 공개 설명에서 no_vocal은 real 메타데이터 필드이며, fake 쪽까지 동일하게 판정하는 필드로 제시되지 않는다. 전체곡에 보컬이 있어도 5초 crop은 instrumental 구간일 수 있다. 당시 분석 CSV와 실제 학습 crop 목록 없이 모델이 반주를 전혀 보지 못했다고 결론 낼 수 없다. [공식 메타데이터 설명](https://github.com/awsaf49/sonics#-metadata-properties).

**AntiDeepfake XLS-R-1B — 연구팀 공식 공개 모델 사용 기록, 직접 학습 없음**

`nii-yamagishilab/xls-r-1b-anti-deepfake`는 XLS-R-1B 특징 추출기와 평균 pooling·FC 분류기로 이루어진 발화 위조 탐지 모델이다. 공개 카드상 약 56,370시간 real + 18,280시간 fake, 총 74,650시간·100개 이상 언어로 후속학습했다. 실제 다국어 발화, TTS·VC·보코더·코덱 및 확산 모델 기반 합성 발화가 중심이다. **74k시간을 우리가 학습한 것은 아니다.** [연구팀 모델 카드](https://huggingface.co/nii-yamagishilab/xls-r-1b-anti-deepfake).

발화 데이터의 큰 규모와 생성 방식 다양성을 활용하여 Voice 분기의 일반화를 개선하려는 선택이다. 공식 기본판은 RawBoost (1)+(2), cross-entropy, AdamW 설정이지만, 초기 제출 ZIP이 없어 기본판/NDA와 정확한 revision은 재확인하지 못했다. 음악·가창에 특화된 후속학습 모델이라고 표현하지 않는다.

공개 학습 목록: AISHELL3, ASVspoof2019-LA·2021-LA·2021-DF·ASVspoof5, CFAD, CNCeleb2, Codecfake/CodecFake, CVoiceFake, DECRO, DFADD, Diffuse or Confuse, DiffSSD, DSD, FLEURS/FLEURS-R, HABLA, LibriTTS/LibriTTS-R/LibriTTS-Vocoded, LJSpeech, MLAAD, MLS, SpoofCeleb, VoiceMOS, VoxCeleb2/VoxCeleb2-Vocoded, WaveFake. 이는 공개 후속학습 구성이고, 우리의 로컬 다운로드·학습 목록이 아니다.

**DeepFense EAT-large + AASIST — 공개 후속학습 checkpoint, 직접 학습 없음**

후속 ZIP의 정확한 출처는 `DeepFense/FakeMusicCaps_EAT_AASIST_NoAug_Seed2`다. EAT는 음악·일반 오디오 표현을 만들고 AASIST가 진위를 판별하는 backend다. 초기 EAT 표현학습은 `worstchan/EAT-large_epoch20_pretrain`, AS-2M(AudioSet) 사전학습이며, 최종 추론에는 DeepFense가 후속학습한 frontend까지 포함된 가중치를 쓴다. **원본 EAT 사전학습 가중치만 가져다 쓴 것과 다르다.** [EAT 사전학습 카드](https://huggingface.co/worstchan/EAT-large_epoch20_pretrain), [실제 제출 출처 문서](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/windows/submit_eat75_clam25_fmc_v3/model/eat_fmc/EAT_MODEL_INFO.txt).

공개 config에서 16kHz, 64,000 samples(4초), train의 random crop/repeat pad, 증강 목록 없음, seed2, batch16, cross-entropy, Adam lr=1e-6, 최대 100 epoch·early stopping patience7을 확인했다. 최대 epoch 설정은 실제 100 epoch 완료 증거가 아니다. 라벨은 `spoof=0, bonafide=1`이므로 제출 코드도 softmax 0번을 fake로 읽는다. [해당 checkpoint의 학습 config](https://huggingface.co/DeepFense/FakeMusicCaps_EAT_AASIST_NoAug_Seed2/blob/main/config.yaml).

다만 config의 실제 train/dev parquet는 제작자의 로컬 경로만 제시돼 있다. 공개 FakeMusicCaps 데이터 카드의 27,605행은 spoof 목록이다. **배포 EAT가 학습한 정확한 real 파일 출처·개수, 생성기별 subset, train/dev 명단을 그 카드만으로 확정할 수 없다.** 일반 FMC 비교 체계의 real은 MusicCaps지만, 그것을 EAT의 정확한 4,120개 학습 목록과 동일시해서는 안 된다. 4,120개는 우리의 MusicDET 기록이다. [DeepFense FakeMusicCaps 데이터](https://huggingface.co/datasets/DeepFense/FakeMusicCaps).

현재 후속 EAT ZIP은 원본 mix를 16kHz mono로 읽고 **앞 4초**, 짧으면 반복, fake softmax를 사용한다. 추가 로지스틱 보정은 없다. 초기 `music_eat` ZIP의 앞/중앙 crop·집계 방식까지 이것과 같다고 확정할 수 없다. 신경망 학습 대신 가중치를 추론용 state로 정리하고 timm 유틸리티를 옮긴 작업은 패키징이다.

모델 선택 이유는 FMC 기반 음악 탐지라는 관련성과, 이미 0.7224를 얻은 EAT 구성의 실증 결과다. 다만 이는 구조·학습 데이터·입력 설정이 합쳐진 결과로, “EAT 구조만의 상승폭”이나 “FMC 데이터만의 상승폭”은 아니다.

**MusicDET SpecNF — 직접 학습한 탐지기**

사용한 것은 공식 SpecNF 구조의 **log-power spectrogram → CNN → 주파수 대역 normalizing flow**다. real 음악 분포에 대한 NLL을 출력한다. 이 실행은 MERT/XLS-R 표현에 flow를 붙인 모델도, real/fake class-conditional 양쪽을 학습한 모델도 아니다. 공식 repository에 여러 구조가 있다고 해서 모두 이 ZIP에 사용된 것은 아니다. [공식 코드](https://github.com/Chaolei98/MusicDET), [로컬 SpecNF 구현](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/vendor/MusicDET/model.py:646).

직접 학습한 이유는 당시 바로 사용할 수 있는 학습 완료 MusicDET-FMC 탐지기 checkpoint를 확보하지 못했기 때문이다. 정확한 표현은 **“공식 구조를 사용하여 로컬에서 학습”**이다. 현재 공개 가중치가 세계 어디에도 존재하지 않는다는 주장과 구분한다.

| 단계 | 실제 사용한 데이터·가공 |
|---|---|
| real 원천 | MusicCaps의 YouTube ID와 start/end에 해당하는 10초 원본 음악 mix. 보컬 포함 음악과 악기 중심 음악이 모두 가능한 데이터. instrumental-only로 선별하지 않음 |
| fake 원천 | FakeMusicCaps 공개 생성 오디오. MusicGen medium, MusicLDM, AudioLDM2, Stable Audio Open, Mustango의 5종 |
| split | MusicDET 공개 openset TTM01–05 protocol. 생성기별로 같은 real ID split 공유. 원곡 ID 기준 train/dev/eval 교집합 0인 원본 protocol을 유지 |
| 다운로드 | protocol real 5,493개 중 5,147개 확보. MusicCaps 전체 배포 5,521개와 숫자가 다름. 누락된 real은 다른 곡으로 채우지 않고 protocol에서 제외 |
| fake 확보 | protocol에 필요한 27,465개 확보. FMC 전체 배포 27,605개와 구분 |
| 파일 가공 | fake 폴더명을 TTM01–05 prefix로 매핑. stereo→mono float32, 원본 SR→48kHz→16kHz(`soxr_hq`), FLOAT WAV 저장. 총 32,612개 |
| 실제 가중치 학습 | TTM01 train에서 확보한 **real 4,120개만**. 16kHz 64,600 samples. 긴 파일 random crop, 짧으면 무작위 위치 zero-pad, RMS 정규화 |
| 모델 선택 | TTM01 dev의 real/fake를 원곡 ID로 짝지은 510+510개. 방향 `real score=-NLL`을 고정하고 dev EER가 가장 낮은 epoch 선택 |
| 확률 보정 | 5개 생성기의 dev를 합쳐 real 2,550행 + fake 2,745행. real 고유 파일은 510개이며 5회 반복. 총 5,295행, 고유 오디오는 3,255개 |
| 추론 | 원본 mix 중앙 64,600 samples=4.0375초. 짧으면 중앙 zero-pad, RMS 정규화. 기본 center 모드 사용 |

MusicCaps는 전문가가 음악 설명을 단 AudioSet 유래 10초 클립이며, 원본 음악의 성격이 다양하다. 특정 vocal/instrumental 비율은 이번 사용 subset에서 확인되지 않았다. FMC의 음악 생성 라벨도 DACON의 Voice/Music 각각의 라벨은 아니다. **이 프로젝트에서 FR/RF 성분 교차 합성이나 Demucs stem 학습은 하지 않았다.** [MusicCaps 공식 카드](https://huggingface.co/datasets/google/MusicCaps), [FMC 공식 배포](https://zenodo.org/records/15063698).

학습 설정은 K=2, L=1, R=5, seed688, batch16, 10 epoch, Adam lr=5e-4, weight decay=5e-4, gradient norm clipping=100이다. pretrained 탐지기 초기화 없이 SpecNF를 학습했고, **epoch4**가 선택됐다. train의 NLL 평균을 최소화했다. `SpecAugmentFT` 객체는 구조에 정의돼 있지만 해당 SpecNF.forward에서는 호출되지 않아, 이름만 보고 SpecAugment를 적용했다고 기록하지 않는다. [학습 코드](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/train_musicdet_real_only.py), [가공 코드](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/prepare_musicdet_fmc_dataset.py), [학습 로그](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/musicdet_training_log.txt).

4.0375초는 DACON에서 최적화해 얻은 길이가 아니라 공식 구조·학습 조건에 맞춘 값이다. 원본 mix 선택은 직접 학습한 MusicCaps 입력과 비슷하게 유지하고, 분리 과정의 변화를 줄이려는 선택이었다. 필수불가결한 방식은 아니며 반주 입력과 독립 비교가 가능하다. baseline 대비 모델 외에 입력·crop·정규화·확률 변환도 바뀌었다.

**MusicDET의 calibration은 왜 했는가**

NLL은 확률이 아니므로 MUSIC_FAKE_PROB와 FILE 결합에 바로 넣을 수 없다. dev의 NLL 한 개를 설명변수, real=0/fake=1을 정답으로 로지스틱 회귀를 적합했다. `class_weight=balanced`, `lbfgs`, `max_iter=1000`, seed688을 사용했다.

`P(fake) = sigmoid(4.038366120662576 × NLL − 5.510506263478469)`

두 파라미터만 학습하는 단순한 고정 변환이라 데이터가 작은 상황에서 설명·재현하기 쉽다. 양의 기울기이므로 원점수의 순위를 보존하지만, 확률의 크기를 사용하는 FILE MAX에는 영향을 준다. class-balanced 학습과 데이터 분포 차이가 있으므로 DACON의 실제 사후확률로 정확히 보정됐다는 보장은 없다. 클리핑·반올림에 따른 동점도 순위 지표에 작은 차이를 만들 수 있다. [보정 코드](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/calibrate_musicdet_fmc.py), [ZIP의 실제 보정값](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/windows/submit_musicdet_fmc_v1/model/musicdet_fmc/calibration.json).

**real-only는 가중치 업데이트 범위의 말이다.** fake는 신경망 업데이트에는 사용하지 않았지만 epoch 선택과 calibration에는 사용했다. 따라서 전체 개발 과정이 fake 미사용인 순수 zero-shot이라고 표현하면 부정확하다. 이번 완료 기록에서 DACON 평가 오디오나 라벨로 gradient·보정기를 학습한 근거는 없다. 리더보드 결과를 보고 후속 실험 방향을 정한 것은 별개다.

**MoM-CLAM — 공식 가중치 + 버전에 따라 직접 보정**

공식 checkpoint는 `best_model_triplet_loss_margin_0.2.pth`, 저장소 commit `db7fff06e25415c119e58b8e4c4f48c840a845c8`다. 기존 검증 manifest에서 공개 원본과 제출 파일의 SHA-256 일치를 확인했다: `01a43bb0c592ac5ba50e9689c65f0c5e0a77d287d52e17cdd35f61ff684deab6`. **모든 CLAM 버전에서 추가 신경망 학습은 없다.** [출처·해시 manifest](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/clam_official_v1_manifest.json).

CLAM은 같은 원본 mix를 MERT와 Wav2Vec2 두 encoder에 넣고, 층별 특징을 집계·attention으로 결합해 곡 진위 점수 하나를 출력한다. MoM이 데이터셋이고 CLAM이 모델이다. 두 encoder가 실제 분리된 보컬·반주를 각각 입력받는 구조가 아니다. 공개 학습은 BCE와 triplet loss를 결합한다. [CLAM 논문](https://arxiv.org/html/2512.00621v1), [고정 commit 학습 코드](https://github.com/StarkVision-AI/MoM-CLAM/blob/db7fff06e25415c119e58b8e4c4f48c840a845c8/triplet_train.py).

| 학습 층위 | 데이터·성격 |
|---|---|
| MERT-v1-95M 사전학습 | 모델 카드상 음악 20,000시간, 24kHz, 5초 문맥의 음악 표현학습. 구체적 전체 파일 목록 미확인. 330M 모델의 160,000시간과 구분 |
| Wav2Vec2 branch | `m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres`. GTZAN 음악 장르 분류로 조정된 공개 모델의 hidden state 사용. 순수 발화 detector나 보컬 전용 모델로 단정하면 안 됨 |
| MoM real | 실제 원곡 및 사람이 부른 커버. 원본 mix/노래 단위 real |
| MoM fake train/dev | Suno v2·v3.5, Udio v1.5, DiffRhythm 생성 노래 |
| MoM test 전용 fake | Riffusion, Suno v3·v4, YuE, AI voice covers. 이 목록을 학습 데이터에 넣었다고 쓰면 안 됨 |

MERT와 Wav2Vec2의 사전학습은 우리가 실행한 학습이 아니다. GTZAN은 음악 장르 분류 데이터이지 성분 진위 데이터가 아니다. [MERT 모델 카드](https://huggingface.co/m-a-p/MERT-v1-95M), [Wav2Vec2-GTZAN 모델 카드](https://huggingface.co/m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres).

DiffRhythm의 Mostly Fake는 실제 가사를 조건으로 오디오를 생성한 경우다. **실제 반주에 가짜 보컬만 얹었다는 뜻이 아니다.** 공개 학습 실행 로그의 real 48,268개 + fake 47,901개 후보와 80:20 분할 기록은 있지만, 현 공개 CSV와 당시 파일 수 차이·누락이 있어 배포 checkpoint의 정확한 파일별 학습 이력은 미확인이다. instrumental-only가 몇 개인지도 확정하지 않는다. [공개 학습 노트북](https://github.com/StarkVision-AI/MoM-CLAM/blob/db7fff06e25415c119e58b8e4c4f48c840a845c8/train_triplet_loss.ipynb).

선택 이유는 음악 특화 MERT와 다른 표현을 결합하고, FMC와 다른 MoM 생성기들의 학습 신호가 EAT를 보완할 수 있다는 가설이었다. 서로 다른 데이터·구조는 다양성의 근거지만, **실제로 오류를 상호 보완한다는 증거는 별도로 필요하다.** 현재 제출 점수에서는 CLAM 추가가 기존 EAT 기록을 넘는 이득으로 이어지지 않았다.

**5. CLAM 버전 차이와 보정의 타당성**

| 항목 | EAT+CLAM v1/v2 | CLAM 단독 v1 | CLAM 단독 v2 | EAT+CLAM v3 |
|---|---|---|---|---|
| CLAM 가중치 | 공식 동일 | 공식 동일 | 공식 동일 | 공식 동일 |
| head batch | 1 | 1 | 16 | 16 |
| input | 원본 mix→librosa 16kHz→MERT24kHz/W2V16kHz | native decode→각 SR | native decode→각 SR | 앙상블 v2 경로 유지 |
| 길이 | 첫 90초, 뒤 zero-pad | 동일 길이 규칙 | 동일 길이 규칙 | 동일 길이 규칙 |
| 점수 | 음수 기울기 FMC 보정 | 원래 sigmoid, float64 계산 | 원래 sigmoid, float32 | 새 양수 기울기 FMC 보정 |
| Music 최종 | EAT75:CLAM25 logit | CLAM | CLAM | EAT75:CLAM25 logit |
| FILE에 들어가는 MF | 앙상블 MF | CLAM MF | CLAM MF | 앙상블 MF |
| 직접 학습 | 보정기 | 없음 | 없음 | 보정기 재적합 |

공식 head는 `batch_first=False` attention에 `[B,1,768]`을 입력해, 같은 배치의 다른 파일을 sequence처럼 참조한다. v1의 B=1과 공식 학습·평가의 B=16이 다르게 작동한다. 배치16 버전은 ID를 정렬한 뒤 별도 torch generator seed42로 그룹을 고정하고 잔여 배치를 보존한다. **공식 동작 재현이며, 파일 간 의존성을 제거한 버그 수정은 아니다.** 원래 공식 평가 당시의 정확한 파일 묶음까지 재현했다고 할 수 없다. [배치 계약 검사](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/clam_v2_contract_audit.json), [공식 head](https://github.com/StarkVision-AI/MoM-CLAM/blob/db7fff06e25415c119e58b8e4c4f48c840a845c8/models/clam.py).

CLAM v1/v2 앙상블 보정은 TTM01 dev real100+MusicGen fake100에 맞춘 다음 식이다.

`P_CLAM = sigmoid(−0.8298080266953047 × raw_logit + 9.82523342546405)`

음수 기울기는 단순한 범위 조정에 더해 **순위 방향을 뒤집는다**. 공식 라벨을 반대로 읽은 것이 아니라, 당시 FMC·배치1 출력에서 역방향 관계가 관측돼 그렇게 적합한 것이다. 공식 fake 라벨은 1이고 sigmoid(raw_logit)가 원래 방향이다. 짧은 FMC 클립을 90초로 채우는 조건과 배치 동작을 충분히 분리하지 않은 상태에서 이 관계를 DACON으로 옮긴 것은 근거가 약했다.

v3는 동일 TTM01 dev real100+fake100에서 새 배치 조건의 점수로 재적합했다.

`P_CLAM_v3 = sigmoid(0.4315278838792132 × raw_logit − 6.293195255530049)`

이제 기울기가 양수다. 배치 조건이 바뀌면 출력 분포가 바뀌므로 보정을 다시 맞추는 이유는 타당하지만, **배치 변경과 보정 재학습의 개별 효과를 분리하는 실험은 아니다.** v3의 75:25·seed는 새 eval에서 최적화하지 않았으나, 앞선 v1/v2 분석 코드는 여러 가중치를 dev와 eval 양쪽에서 확인했다. 따라서 전체 탐색 과정을 일관되게 “eval을 전혀 보지 않은 가중치 선택”으로 쓰면 안 된다. [초기 분석 코드](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/analyze_clam_eat.py), [v3 실제 보정값](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/windows/submit_eat75_clam25_fmc_v3/model/clam/fmc_batch16_calibration.json).

CLAM 단독 v2와 EAT+CLAM v3는 입력 디코더·경로와 보정이 다르므로, 둘의 점수 차이를 EAT 결합 효과만으로 해석하면 안 된다. 다만 DACON 공식 데이터는 이미 16kHz다. 그러므로 native→24kHz 경로가 DACON에서 잃어버린 고주파를 복원했다는 해석도 성립하지 않는다. 코드 차이와 실제 성능 기여는 구분해야 한다.

추가로 FMC dev/eval끼리 파일 중복이 없다는 것은 확인됐지만, 공개 EAT checkpoint 제작자의 정확한 train/dev 목록과 교차 대조한 것은 아니다. EAT의 FMC 평가를 checkpoint 학습과 완전히 독립된 새 데이터라고 확정할 수 없다. 이번 점검에서는 이러한 로컬 점수로 제출 성능을 주장하지 않는다.

**6. EAT+MusicDET와 EAT+CLAM은 FILE 연결이 다르다**

| 구성 | MUSIC_FAKE_PROB | FILE_FAKE_PROB |
|---|---|---|
| MusicDET 단독 | 보정된 MusicDET | `max(VP×VF, MP×MusicDET)` |
| EAT75+MusicDET25 | `sigmoid(0.75×logit(EAT)+0.25×logit(MusicDET))` | **`max(VP×VF, MP×EAT)`** |
| EAT75+CLAM25 | `sigmoid(0.75×logit(EAT)+0.25×logit(보정 CLAM))` | **`max(VP×VF, MP×앙상블)`** |

이 결합은 확률 75:25 산술평균이 아니라 logit 평균이다. 양쪽 확률은 1e-5~1−1e-5로 제한한다. 가중치 25%가 실제 영향력도 항상 25%라는 뜻은 아니다. 각 점수의 logit 크기와 보정에 따라 순위·확률 변화가 달라진다.

EAT+MusicDET의 이유는 MusicDET의 보완 신호를 MUSIC 컬럼에만 적용하고, 기존에 유리했던 EAT 기반 FILE 구성을 보존해 보려는 것이다. **“기존 제출 CSV를 그대로 복사”한 것은 아니고, 현재 코드에서 EAT-only FILE을 다시 계산한다.** 초기 `music_eat` 코드와 완전 동일성이 미확인이므로 과거 FILE 값을 bitwise 유지한다고 표현하지 않는다. 점수가 없는 수정판에서 앙상블 성능 개선을 주장할 근거는 없다. [ZIP에서 추출한 EAT+MusicDET 코드](C:/Users/husky_gp7j99y/Documents/ChatGPT/DACON/submission_audit/windows/submit_eat75_musicdet25/script.py:561).

EAT+CLAM은 새로운 Music 판단을 FILE에도 반영한다. 이 차이 때문에 두 앙상블은 보조 모델만 바꾼 통제 비교가 아니다. 같은 Music EER라도 FILE은 존재 확률·Voice와 결합하며 파일 순위가 다시 만들어져 성능이 달라질 수 있다.

**7. EER 역산과 과거 메모의 수정**

공식 산식은 `Score=0.9×ADS+0.1×CPS`, `ADS=0.5×(1−FileEER)+0.2×(1−VoiceEER)+0.3×(1−MusicEER)`다. Voice/Music EER은 각 성분이 존재하는 파일에서만 평가한다. 따라서 전체 파일에서 잘 작동하는 VF라도 Music 평가 subset에서 같은 의미를 보장하지 않는다. [DACON 공식 평가](https://dacon.io/competitions/official/236749/overview/evaluation).

MusicDET 원본과 FILE=0.5 probe는 ZIP 비교에서 script 외 모든 entry의 CRC·크기가 같다. FILE 외 출력이 동일하다는 조건에서:

`FileEER = 0.5 − (0.6798968254 − 0.591968254)/0.5 = 0.3241428572`

baseline VoiceEER≈0.2244가 유지됐다는 조건을 추가하면:

`MusicEER = 1 − [0.591968254 − 0.25 − 0.2×(1−0.2244)]/0.3 ≈ 0.3771724867`

따라서 과거 baseline 추정값 File≈0.3324, Music≈0.4371과 비교하면 File 약 0.83%p, Music 약 5.99%p 감소다. Voice의 반올림 오차와 실행별 출력 변동 가능성 때문에 조건부 역산값으로 표기한다. EAT+CLAM·CLAM 단독은 총점과 ADS만으로 File/Music EER을 각각 유일하게 구할 수 없다.

다음 과거 표현은 수정해야 한다.

| 과거 표현 | 이번 점검의 판단 |
|---|---|
| SONICS의 EER≈0.5이므로 정보가 전혀 없다 | 해당 제출 설정이 부진했다. 다른 threshold·조건·subset의 구분 신호까지 부정할 근거는 없음 |
| no_vocal=False인 전체곡이므로 instrumental을 한 번도 못 봤다 | 전체곡 보컬 유무와 crop의 보컬 유무는 다름. 실제 CSV·학습 crop 확인 필요 |
| Voice mix 제출 하락으로 입력 문제 가능성이 사라졌다 | 확인한 한 가지 입력 변경만 불리했다. 분리 품질·crop·집계·정규화 등은 남음 |
| Music=Voice 제출 실패로 평가셋에 real music+fake voice가 많음이 증명됐다 | 성분별 분포를 총점 하나로 식별할 수 없음. Voice 점수의 Music 직접 대체가 실패했다는 결과 |
| CPS가 높으므로 분리된 두 성분의 진위도 독립적으로 판정된다 | CPS는 존재 ROC-AUC다. stem 누설과 진위 오염을 검증하지 않음 |
| FILE 음악 항 제거 시 File EER 0.3324→0.3383은 개선 | EER 증가 **+0.0059**, 약 0.59%p 악화다 |
| AntiDeepfake는 FILE 고정이고 VoiceEER≈0.208 | 두 주장과 총점 +0.0067은 동시에 맞지 않음. VF만 0.2244→0.208이면 총점 증가는 약 0.002952. 원본 FILE 고정 여부·역산식을 재확인해야 함 |
| SONICS 0.6690의 MusicEER=0.5086 확정 | 초기 점수·기준값과 FILE 고정 가정만으로 재산출하면 약 0.5182. 6번도 같은 주의가 필요. 기존 0.5086/0.5171은 검증되지 않은 기록값으로 취급 |
| EAT+CLAM v3 − CLAM 단독 v2 = EAT만 추가한 효과 | 입력·보정이 함께 다르므로 불가 |
| 공식 checkpoint 사용이면 공식 평가를 그대로 재현 | 추가 전처리·배치·보정·결합·평가 데이터 차이를 별도로 확인해야 함 |

초기 2·3번 상수 probe로 얻었다는 baseline File/Voice/Music EER도 원본 ZIP과 정확한 원시 제출 수치가 없어 이번에는 “기존 기록의 추정 기준값”으로 두었다. 점수 하나로 여러 EER을 채우지 않는다.

**현재 판단**

가장 근거 있는 음악 분기 개선은 EAT 제출 구성(약 0.7224)이며, 직접 학습 MusicDET 구성도 baseline 대비 개선됐다. CLAM 배치 재현과 보정 재적합은 이전 CLAM 버전보다 점수가 높았지만 기존 EAT 기록을 넘지 못했다. 파일 간 attention을 유지한 CLAM 단독 v2·앙상블 v3는 위 파일 독립 예측 규정에 따라 최종 후보에서 제외해야 한다.

남은 핵심 불확실성은 초기 ZIP의 정확한 전처리·FILE 연결, 공개 EAT/CLAM의 파일별 학습 이력, 전체곡 진위와 DACON 성분 진위의 차이다. 공식 모델의 존재만으로 이 차이가 해결되지는 않는다.

다음 비교의 기준은 **같은 EAT 코드·입력·보정·FILE 연결을 고정하고, 보조 모델 결합 여부 한 가지를 바꾸는 것**이다. 새로운 학습이나 데이터 합성 계획은 이번에 실행·제출한 기법으로 기록하지 않는다.
