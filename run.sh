#!/bin/bash
# run_server.sh

# 가상환경이 없으면 생성
if [ ! -d "venv" ]; then
    echo "가상환경을 생성합니다..."
    python3 -m venv venv || { echo "가상환경 생성 실패"; exit 1; }
fi

echo "가상환경 활성화 중..."
source venv/bin/activate

echo "pip, setuptools, wheel 업그레이드 중..."
pip install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
    echo "종속성 설치 중..."
    pip install -r requirements.txt
else
    echo "requirements.txt 파일이 없습니다. 필요한 라이브러리를 직접 설치하세요."
fi

echo "Flask 서버 실행 중..."
python app.py