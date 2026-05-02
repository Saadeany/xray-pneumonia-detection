Step 1: Install
pip install -r requirements.txt


Step 2: Train Model
cd src
python train.py


Step 3: Run API
cd ../api
python -m uvicorn app:app --reload

Open:
http://localhost:8000/docs



4.docker setup

docker compose build
 
docker compose up

test
python tests/test_api.py


tests
docker run -d --name test-ci -p 8001:8000 -e MODEL_TEST_MODE=1 xray-pneumonia-api
Start-Sleep 15
curl http://localhost:8001/health
docker rm -f test-ci

