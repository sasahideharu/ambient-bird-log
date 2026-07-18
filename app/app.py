import streamlit as st
import pandas as pd
import os
import io
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from scipy.io import wavfile

# --- 1. 環境変数とSupabase接続 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../.env"))

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "bird-wav"

if not url or not key:
    st.error("⚠️ .envファイルからSupabaseのAPIキーが見つからないぜ。")
    st.stop()

supabase: Client = create_client(url, key)

# --- 2. リモート音声の取得と切り出し ---
@st.cache_data(show_spinner=False)
def get_sliced_remote_wav(file_url, original_start, original_end):
    response = requests.get(file_url)
    if response.status_code != 200:
        raise Exception("ファイルのダウンロードに失敗したぜ。")
        
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

# --- 2.5. 日付情報の自動抽出と曜日の計算 ---
def extract_date(filename):
    match = re.match(r'^(\d{6})', filename)
    if match:
        try:
            dt = datetime.strptime(match.group(1), '%y%m%d')
            # 曜日を日本語で取得するロジックを追加
            weekdays = ['月', '火', '水', '木', '金', '土', '日']
            weekday_str = weekdays[dt.weekday()]
            return f"{dt.strftime('%Y/%m/%d')} ({weekday_str})"
        except ValueError:
            return "不明な日付"
    return "不明な日付"

# --- 2.8. 画面遷移のコントロール（Session State） ---
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = None
if 'selected_bird' not in st.session_state:
    st.session_state.selected_bird = None

def go_to_date_detail(date_str):
    st.session_state.page = 'date_detail'
    st.session_state.selected_date = date_str

def go_to_bird_detail(bird_name):
    st.session_state.page = 'bird_detail'
    st.session_state.selected_bird = bird_name

def go_to_main():
    st.session_state.page = 'main'
    st.session_state.selected_date = None
    st.session_state.selected_bird = None

# --- 3. メインUI ---
st.title("🎧 Ambient Bird Log 🐦")

try:
    # アプリ全体で使うために全データを一括取得してPandasへ
    response_all = supabase.table("detections").select("*").execute()
    
    if response_all.data:
        df_all = pd.DataFrame(response_all.data)
        df_all['record_date'] = df_all['wav_filename'].apply(extract_date)
        
        # --- A. メインページ（タブ表示） ---
        if st.session_state.page == 'main':
            tab_bird, tab_location = st.tabs(["🐦 鳥から探す", "📍 場所から探す"])

            # 【タブ1：鳥から探す（Page 1）】
            with tab_bird:
                st.markdown("### 🔍 和名で検索")
                
                # 登場が多い鳥順にカウントして並び替え
                bird_counts = df_all['common_name'].value_counts()
                
                # 検索窓（オプションとして残す）
                search_query = st.text_input("鳥の名前を入力...", placeholder="例：シジュウカラ")
                
                st.markdown("**🦆 検出リスト（登場回数順）**")
                # ボタンを並べてリスト化
                for bird_name, count in bird_counts.items():
                    if not search_query or search_query in bird_name:
                        # ボタンを押すと「鳥から探すページ1-2」へ遷移
                        st.button(
                            f"{bird_name} ({count}件)", 
                            key=f"btn_bird_{bird_name}", 
                            on_click=go_to_bird_detail, 
                            args=(bird_name,),
                            use_container_width=True
                        )

            # 【タブ2：場所から探す（Page 2）】
            with tab_location:
                st.markdown("### 🗺️ 場所から探す")
                df_loc = df_all.dropna(subset=['latitude', 'longitude'])
                
                if not df_loc.empty:
                    df_map = df_loc[['latitude', 'longitude', 'location_name']].drop_duplicates(subset=['latitude', 'longitude'])
                    st.map(df_map, latitude='latitude', longitude='longitude', color='#39FF14', size=150)
                    st.divider()
                    
                    st.markdown("**📍 過去の記録（場所別）**")
                    unique_locations = sorted(df_loc['location_name'].unique().tolist())
                    selected_loc = st.selectbox("場所を選択してくれ:", unique_locations, label_visibility="collapsed")
                    
                    if selected_loc:
                        loc_filtered = df_loc[df_loc['location_name'] == selected_loc].sort_values(by='record_date', ascending=False)
                        dates_at_loc = loc_filtered['record_date'].unique()
                        
                        for date_str in dates_at_loc:
                            count = len(loc_filtered[loc_filtered['record_date'] == date_str])
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"### 📅 {date_str}")
                                    st.caption(f"🎧 検出: {count} 件")
                                with col2:
                                    st.button("詳細", on_click=go_to_date_detail, args=(date_str,), key=f"btn_{selected_loc}_{date_str}")
                            st.divider()
                else:
                    st.info("地図に表示できるデータがないぜ。")

        # --- B. 鳥の詳細ページ (Page 1-2) ---
        elif st.session_state.page == 'bird_detail':
            st.button("⬅️ メインに戻る", on_click=go_to_main)
            
            target_bird = st.session_state.selected_bird
            bird_data = df_all[df_all['common_name'] == target_bird].sort_values(by='confidence', ascending=False)
            
            if not bird_data.empty:
                scientific_name = bird_data.iloc[0]['scientific_name']
                st.markdown(f"## 🐦 {target_bird}")
                st.caption(f"学名: *{scientific_name}*")
                
                # 画像のプレースホルダー（複数枚スライドの準備地）
                st.info("🖼️ **Next Phase:** ここに鳥の画像（スライド）が表示される予定だ。")
                st.divider()
                
                for index, row in bird_data.iterrows():
                    with st.container():
                        wav_filename = row['wav_filename']
                        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                        duration = round(float(row['end_sec']) - float(row['start_sec']), 1)
                        confidence_pct = int(row['confidence'] * 100)
                        
                        # ハイパーリンク風のメタデータ表示
                        col_meta1, col_meta2 = st.columns([1, 1])
                        with col_meta1:
                            st.markdown(f"**信頼度:** `{confidence_pct}%`")
                            st.markdown(f"**再生時間:** `{duration}秒`")
                        with col_meta2:
                            st.button(f"📅 {row['record_date']}", on_click=go_to_date_detail, args=(row['record_date'],), key=f"link_date_{index}")
                            loc_name = row['location_name'] if pd.notna(row['location_name']) else "場所不明"
                            st.button(f"📍 {loc_name}", key=f"link_loc_{index}") # 場所詳細ページへの布石
                        
                        # プレイヤー
                        try:
                            with st.spinner("Loading..."):
                                sliced_audio, actual_start, actual_end = get_sliced_remote_wav(
                                    public_url, float(row['start_sec']), float(row['end_sec'])
                                )
                            st.audio(sliced_audio, format="audio/wav")
                        except Exception as e:
                            st.error(f"ロードエラー: {e}")
                    st.divider()

        # --- C. 記録日詳細ページ (Page 3) ---
        elif st.session_state.page == 'date_detail':
            st.button("⬅️ メインに戻る", on_click=go_to_main)
            
            target_date = st.session_state.selected_date
            st.markdown(f"## 📅 {target_date} の記録")
            
            day_data = df_all[df_all['record_date'] == target_date]
            
            if not day_data.empty:
                visited_locations = day_data['location_name'].dropna().unique()
                loc_text = "、".join(visited_locations) if len(visited_locations) > 0 else "場所不明"
                st.info(f"📍 **その日に行った場所:** {loc_text}")
                
                st.markdown(f"### 🐦 その日の鳥 (計 {len(day_data)} 件)")
                day_data_sorted = day_data.sort_values(by='confidence', ascending=False)
                
                for index, row in day_data_sorted.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 2])
                        wav_filename = row['wav_filename']
                        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                        
                        with col1:
                            confidence_pct = int(row['confidence'] * 100)
                            st.button(f"**{row['common_name']}**", on_click=go_to_bird_detail, args=(row['common_name'],), key=f"link_bird_{index}")
                            st.progress(row['confidence'], text=f"信頼度: {confidence_pct}%")
                        with col2:
                            try:
                                with st.spinner("Loading..."):
                                    sliced_audio, actual_start, actual_end = get_sliced_remote_wav(
                                        public_url, float(row['start_sec']), float(row['end_sec'])
                                    )
                                st.audio(sliced_audio, format="audio/wav")
                            except Exception as e:
                                st.error(f"ロードエラー: {e}")
                    st.divider()

    else:
        st.warning("クラウドのデータベースにデータがないぜ。")
        
except Exception as e:
    st.error(f"システムエラー: {e}")