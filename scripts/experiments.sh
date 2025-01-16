#!/bin/bash

# experiments.sh

# 작업 디렉토리 기준 경로 설정
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_BASH_DIR="${BASE_DIR}/train_bash"
LOG_DIR="../experiments/logs"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 실험 목록 정의 (train.sh, train2.sh, ...)
TRAIN_SCRIPTS=("train1.sh" "train2.sh" "train3.sh" "train4.sh")  # 필요한 만큼 추가

# 실험 이름 목록 정의 (각 train.sh에 대응)
EXPERIMENT_NAMES=("0116_1" "0116_2" "0116_3" "0116_4")  # 필요한 만큼 추가

# 실험 개수 확인
NUM_EXPERIMENTS=${#TRAIN_SCRIPTS[@]}
if [ ${#EXPERIMENT_NAMES[@]} -ne $NUM_EXPERIMENTS ]; then
    echo "실험 스크립트와 실험 이름의 개수가 일치하지 않습니다."
    exit 1
fi

# 각 실험 스크립트를 순차적으로 실행
for ((i=0; i<NUM_EXPERIMENTS; i++)); do
    TRAIN_SCRIPT="${TRAIN_SCRIPTS[$i]}"
    EXPERIMENT_NAME="${EXPERIMENT_NAMES[$i]}"
    TRAIN_SCRIPT_PATH="${TRAIN_BASH_DIR}/${TRAIN_SCRIPT}"
    LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"

    echo "실험 ${EXPERIMENT_NAME} 시작: ${TRAIN_SCRIPT}"
    echo "로그 파일: ${LOG_FILE}"

    # 스크립트 존재 여부 확인
    if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
        echo "스크립트 파일이 존재하지 않습니다: $TRAIN_SCRIPT_PATH"
        exit 1
    fi

    # 스크립트 실행 및 로그 저장
    bash "$TRAIN_SCRIPT_PATH" "$EXPERIMENT_NAME" > "$LOG_FILE" 2>&1

    # 종료 상태 확인
    if [ $? -ne 0 ]; then
        echo "실험 ${EXPERIMENT_NAME} 실패. 시퀀스를 중단합니다."
        exit 1
    fi

    echo "실험 ${EXPERIMENT_NAME} 완료."
done

echo "모든 실험이 성공적으로 완료되었습니다."
