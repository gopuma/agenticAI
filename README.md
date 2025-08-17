# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your keys

# 3. Configure AWS
aws configure

# 4. Launch application
python run.py
# or directly: streamlit run app.py
