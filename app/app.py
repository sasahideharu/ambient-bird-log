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

def go_to_detail(date_str):
    """詳細ページへ切り替える関数"""
    st.session_state.page = 'detail'
    st.session_state.selected_date = date_str

def go_to_main():
    """メインページへ戻る関数"""
    st.session_state.page = 'main'
    st.session_state.selected_date = None

# --- 3. メインUI ---
st.title("🎧 Ambient Bird Log 🐦")

try:
    response_birds = supabase.table("detections").select("common_name").execute()
    
    if response_birds.data:
        # --- A. メインページ（タブ表示） ---
        if st.session_state.page == 'main':
            bird_names = sorted(list(set([row['common_name'] for row in response_birds.data if row['common_name']])))
            tab_bird, tab_location = st.tabs(["🐦 鳥から探す", "📍 場所から探す"])

            # 【タブ1：鳥から探す】
            with tab_bird:
                st.markdown("### 和名で検索")
                selected_bird = st.selectbox("鳥の和名を選択してくれ:", bird_names, label_visibility="collapsed")

                if selected_bird:
                    response_data = supabase.table("detections")\
                        .select("*")\
                        .eq("common_name", selected_bird)\
                        .order("confidence", desc=True)\
                        .execute()
                        
                    filtered_data = response_data.data
                    st.markdown(f"**{selected_bird} の検出リスト (計 {len(filtered_data)} 件)**")
                    
                    for row in filtered_data:
                        with st.container():
                            col1, col2 = st.columns([3, 2])
                            wav_filename = row['wav_filename']
                            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                            
                            with col1:
                                confidence_pct = int(row['confidence'] * 100)
                                record_date = extract_date(wav_filename)
                                st.markdown(f"**{selected_bird}** / `{row['scientific_name']}`")
                                st.caption(f"📅 **記録日:** {record_date} | ⏱️ 検出区間: {row['start_sec']}s 〜 {row['end_sec']}s")
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

            # 【タブ2：場所から探す】
            with tab_location:
                st.markdown("### 🗺️ 場所から探す")
                response_loc = supabase.table("detections")\
                    .select("wav_filename, location_name, latitude, longitude")\
                    .not_.is_("latitude", "null")\
                    .execute()
                
                if response_loc.data:
                    df_loc = pd.DataFrame(response_loc.data)
                    df_loc['record_date'] = df_loc['wav_filename'].apply(extract_date)
                    
                    df_map = df_loc[['latitude', 'longitude', 'location_name']].drop_duplicates(subset=['latitude', 'longitude'])
                    st.markdown("**🌎 フィールドマップ**")
                    st.map(df_map, latitude='latitude', longitude='longitude', color='#39FF14', size=150)
                    st.divider()
                    
                    st.markdown("**📍 過去の記録（場所別）**")
                    unique_locations = sorted(df_loc['location_name'].dropna().unique().tolist())
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
                                    st.caption(f"🎧 検出された野鳥のBeat: {count} 件")
                                with col2:
                                    # ボタンを押すと Session State が切り替わり、詳細ページへ飛ぶ
                                    st.button("詳細を見る", on_click=go_to_detail, args=(date_str,), key=f"btn_{selected_loc}_{date_str}")
                            st.divider()
                else:
                    st.info("まだ地図に表示できる位置情報データがないぜ。")

        # --- B. 記録日詳細ページ ---
        elif st.session_state.page == 'detail':
            # 戻るボタン
            st.button("⬅️ メインに戻る", on_click=go_to_main)
            
            target_date = st.session_state.selected_date
            st.markdown(f"## 📅 {target_date} の記録")
            
            # 全データを取得し、Python側で日付フィルタリングを行う
            response_all = supabase.table("detections").select("*").execute()
            df_all = pd.DataFrame(response_all.data)
            df_all['record_date'] = df_all['wav_filename'].apply(extract_date)
            
            # 選択された日のデータのみを抽出
            day_data = df_all[df_all['record_date'] == target_date]
            
            if not day_data.empty:
                # その日に行った場所を抽出
                visited_locations = day_data['location_name'].dropna().unique()
                loc_text = "、".join(visited_locations) if len(visited_locations) > 0 else "場所不明"
                st.info(f"📍 **その日に行った場所:** {loc_text}")
                
                st.markdown(f"### 🐦 その日の鳥 (計 {len(day_data)} 件)")
                
                # 信頼度が高い順にソートしてカードを表示
                day_data_sorted = day_data.sort_values(by='confidence', ascending=False)
                
                for index, row in day_data_sorted.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 2])
                        wav_filename = row['wav_filename']
                        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                        
                        with col1:
                            confidence_pct = int(row['confidence'] * 100)
                            st.markdown(f"**{row['common_name']}** / `{row['scientific_name']}`")
                            st.progress(row['confidence'], text=f"信頼度: {confidence_pct}%")
                            st.caption(f"⏱️ 検出区間: {row['start_sec']}s 〜 {row['end_sec']}s")
                        
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
                st.warning("この日のデータは見つからなかったぜ。")

    else:
        st.warning("クラウドのデータベースにデータがないぜ。")
        
except Exception as e:
    st.error(f"システムエラー: {e}")