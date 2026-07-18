import streamlit as st
import pandas as pd
import os
import io
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from scipy.io import wavfile

# --- 1. 環境変数とSupabase接続 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../.env"))

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")  # アプリ側は必ず安全なanonキーを使用
BUCKET_NAME = "bird-wav"

if not url or not key:
    st.error("⚠️ .envファイルからSupabaseのAPIキーが見つからないぜ。")
    st.stop()

supabase: Client = create_client(url, key)

# --- 2. リモート音声の取得と切り出し ---
@st.cache_data(show_spinner=False)
def get_sliced_remote_wav(file_url, original_start, original_end):
    """CloudからWAVをダウンロードしてメモリ上で切り出す"""
    response = requests.get(file_url)
    if response.status_code != 200:
        raise Exception("ファイルのダウンロードに失敗したぜ。")
        
    # メモリ上でバイナリデータをWAVとして読み込む
    wav_io = io.BytesIO(response.content)
    sample_rate, data = wavfile.read(wav_io)
    total_duration = len(data) / sample_rate
    
    start_sec = max(0.0, original_start - 1.5)
    end_sec = min(total_duration, original_end + 1.5)
    
    start_frame = int(start_sec * sample_rate)
    end_frame = int(end_sec * sample_rate)
    
    sliced_data = data[start_frame:end_frame]
    
    out_wav = io.BytesIO()
    wavfile.write(out_wav, sample_rate, sliced_data)
    out_wav.seek(0)
    
    return out_wav, start_sec, end_sec

# --- 3. メインUI ---
st.title("BirdNET Audio Player 🐦🎧 (Full Cloud Version)")

try:
    response_birds = supabase.table("detections").select("common_name").execute()
    
    if response_birds.data:
        bird_names = sorted(list(set([row['common_name'] for row in response_birds.data if row['common_name']])))
        selected_bird = st.selectbox("鳥の日本語名を選択:", bird_names)

        if selected_bird:
            response_data = supabase.table("detections")\
                .select("*")\
                .eq("common_name", selected_bird)\
                .order("confidence", desc=True)\
                .execute()
                
            filtered_data = response_data.data
            st.write(f"### {selected_bird} の検出リスト (計 {len(filtered_data)} 件)")
            
            for row in filtered_data:
                col1, col2 = st.columns([1, 1])
                wav_filename = row['wav_filename']
                
                # Cloud Storageから誰でもアクセスできるPublic URLを自動生成
                public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                
                with col1:
                    st.write(f"**ファイル:** `{wav_filename}`")
                    st.write(f"**適合率:** `{row['confidence']:.4f}`")
                
                with col2:
                    try:
                        with st.spinner("Cloudから音源をロード中..."):
                            sliced_audio, actual_start, actual_end = get_sliced_remote_wav(
                                public_url, float(row['start_sec']), float(row['end_sec'])
                            )
                        st.caption(f"🎧 再生区間: {actual_start:.1f}s 〜 {actual_end:.1f}s")
                        st.audio(sliced_audio, format="audio/wav")
                    except Exception as e:
                        st.error(f"音声ロードエラー: {e}")
                
                st.divider()
    else:
        st.warning("DBにデータがないぜ。")
except Exception as e:
    st.error(f"システムエラー: {e}")